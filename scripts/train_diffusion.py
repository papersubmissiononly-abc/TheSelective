import logging
import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
import seaborn as sns
# sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.abspath('./'))
import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans

from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from datasets.pl_pair_dataset import MultiProteinPairedDataset # Import the new dataset
from models.molopt_score_model import ScorePosNet3D


def get_auroc(logger, y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)

def get_pearsonr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--value_only', action='store_true')
    parser.add_argument('--train_report_iter', type=int, default=200)
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()

    # load ckpt
    if args.ckpt:
        print(f'loading {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = ckpt['config']
        # config = misc.load_config(args.config)
    else:
        # Load configs
        config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    
    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.name == 'multipro':
        transform_list.append(trans.NormalizeVinaAndPair(config.data.name))
    else:
        ## Normalize 0~1 안함
        transform_list.append(trans.NormalizeVina(config.data.name))
    
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    if config.data.name == 'pl':
        train_set, val_set, test_set = subsets['train'], subsets['test'], []
    elif config.data.name == 'pdbbind':
        train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    elif config.data.name == 'multipro':
        train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    else:
        raise ValueError

    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)
    # Model
    logger.info('Building model...')
    
    # Copy center_ligand from data config to model config
    if hasattr(config.data, 'center_ligand'):
        config.model.center_ligand = config.data.center_ligand
        logger.info(f'Center ligand mode: {config.data.center_ligand}')
    
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    
    
    # print(model)
    logger.info(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    start_it = 0
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        # start_it = ckpt['iteration']

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                protein_id=batch.protein_id,
                on_target_affinity=batch.on_target_affinity.float(),
                off_target_affinities=batch.off_target_affinities.to(args.device)
            )
            if args.value_only:
                results['loss'] = results['loss_exp']
                
            loss, loss_pos, loss_v, loss_exp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            loss_on = results.get('loss_on_target_affinity', 0)
            loss_off = results.get('loss_off_target_affinity', 0)
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f) | On-Target %.6f | Off-Target %.6f | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_exp, loss_on, loss_off, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
        sum_loss_on_target_exp, sum_loss_off_target_exp = 0, 0
        all_pred_v, all_true_v , all_pred_exp, all_true_exp = [], [], [], []
        all_pred_off_exp, all_true_off_exp = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,

                        protein_id=batch.protein_id,
                        on_target_affinity=batch.on_target_affinity.float(),
                        off_target_affinities=batch.off_target_affinities
                    )
                    loss, loss_pos, loss_v, loss_exp, pred_exp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp'], results['pred_on_exp']
                    loss_on = results.get('loss_on_target_affinity', 0)
                    loss_off = results.get('loss_off_target_affinity', 0)

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_loss_on_target_exp += float(loss_on) * batch_size
                    sum_loss_off_target_exp += float(loss_off) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                    all_pred_exp.append(np.atleast_1d(pred_exp.detach().cpu().numpy()))
                    # Use original batch data instead of double-normalized values
                    all_true_exp.append(np.atleast_1d(batch.on_target_affinity.float().detach().cpu().numpy()))
                    
                    # Add off-target affinity debugging
                    if 'pred_off_exp' in results:
                        all_pred_off_exp.append(results['pred_off_exp'].detach().cpu().numpy())
                        all_true_off_exp.append(batch.off_target_affinities.float().detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n
        avg_loss_on_target_exp = sum_loss_on_target_exp / sum_n
        avg_loss_off_target_exp = sum_loss_off_target_exp / sum_n

        
        # --- 디버깅 코드 추가 시작 ---
        #true_values = np.concatenate(all_true_exp, axis=0)
        #pred_values = np.concatenate(all_pred_exp, axis=0)
        
        #print(f"[Debug ON-TARGET] True values shape: {true_values.shape}")
        #print(f"[Debug ON-TARGET] True values (first 10): {true_values[:10]}")
        #print(f"[Debug ON-TARGET] Predicted values shape: {pred_values.shape}")
        #print(f"[Debug ON-TARGET] Predicted values (first 10): {pred_values[:10]}")
        
        # Off-target debugging
        #if all_pred_off_exp and all_true_off_exp:
        #    true_off_values = np.concatenate(all_true_off_exp, axis=0)
        #    pred_off_values = np.concatenate(all_pred_off_exp, axis=0)
            
        #    print(f"[Debug OFF-TARGET] True values shape: {true_off_values.shape}")
        #    print(f"[Debug OFF-TARGET] True values (first 10): {true_off_values[:10]}")
        #    print(f"[Debug OFF-TARGET] Predicted values shape: {pred_off_values.shape}")
        #    print(f"[Debug OFF-TARGET] Predicted values (first 10): {pred_off_values[:10]}")
        
        # 모든 예측값이 동일한지 확인
        #if np.all(pred_values == pred_values[0]):
        #    logger.warning("All predicted ON-TARGET affinity values are identical! The model might not be learning.")
        # 모든 실제값이 동일한지 확인
        #if np.all(true_values == true_values[0]):
        #    logger.warning("All true ON-TARGET affinity values in this validation batch are identical!")
        # --- 디버깅 코드 추가 끝 ---
        
        atom_auroc = get_auroc(logger, np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        exp_pearsonr = get_pearsonr(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 (On %.6f, Off %.6f) | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, avg_loss_on_target_exp * 1000, avg_loss_off_target_exp * 1000, atom_auroc
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/loss_exp', avg_loss_exp, it)
        writer.add_scalar('val/loss_on_target_affinity', avg_loss_on_target_exp, it)
        writer.add_scalar('val/loss_off_target_affinity', avg_loss_off_target_exp, it)
        writer.add_scalar('val/atom_auroc', atom_auroc, it)
        writer.add_scalar('val/pcc', exp_pearsonr[0], it)
        writer.add_scalar('val/pvalue', exp_pearsonr[1], it)
        # fig = plt.figure(figsize=(12,12))
        
        pred_exp_flat = np.concatenate(all_pred_exp, axis=0).flatten()
        true_exp_flat = np.concatenate(all_true_exp, axis=0).flatten()
        writer.add_figure('val/pcc_fig', sns.lmplot(data=pd.DataFrame({
                'pred': pred_exp_flat,
                'true': true_exp_flat
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(exp_pearsonr[0], exp_pearsonr[1])).fig,it)
        writer.flush()
        
        if args.value_only:
            return avg_loss_exp
        
        return avg_loss

    def test(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
        all_pred_v, all_true_v , all_pred_exp, all_true_exp= [], [], [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Test'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,

                        protein_id=batch.protein_id,
                        on_target_affinity=batch.on_target_affinity.float(),
                        off_target_affinities=batch.off_target_affinities
                    )
                    loss, loss_pos, loss_v, loss_exp, pred_exp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp'], results['pred_on_exp']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                    all_pred_exp.append(np.atleast_1d(pred_exp.detach().cpu().numpy()))
                    #all_true_exp.append(batch.on_target_affinity.float().detach().cpu().numpy())
                    # Use original batch data instead of double-normalized values
                    all_true_exp.append(np.atleast_1d(batch.on_target_affinity.float().detach().cpu().numpy()))
                    
                    # Add off-target affinity debugging for test
                    if 'pred_off_exp' in results:
                        all_pred_off_exp.append(results['pred_off_exp'].detach().cpu().numpy())
                        all_true_off_exp.append(batch.off_target_affinities.float().detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n
        atom_auroc = get_auroc(logger, np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        exp_pearsonr = get_pearsonr(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))
        
        logger.info(
            '[Test] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, atom_auroc
            )
        )
        writer.add_scalar('test/loss', avg_loss, it)
        writer.add_scalar('test/loss_pos', avg_loss_pos, it)
        writer.add_scalar('test/loss_v', avg_loss_v, it)
        writer.add_scalar('test/loss_exp', avg_loss_exp, it)
        writer.add_scalar('test/atom_auroc', atom_auroc, it)
        writer.add_scalar('test/pcc', exp_pearsonr[0], it)
        writer.add_scalar('test/pvalue', exp_pearsonr[1], it)
        # fig = plt.figure(figsize=(12,12))
        
        writer.add_figure('test/pcc_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_exp, axis=0),
                'true': np.concatenate(all_true_exp, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(exp_pearsonr[0], exp_pearsonr[1])).fig,it)
        writer.flush()
        
        if args.value_only:
            return avg_loss_exp
        
        return avg_loss
    
    def cleanup_old_checkpoints(ckpt_dir, current_iter, logger):
        """이전 체크포인트 파일들을 삭제하는 함수"""
        try:
            # 체크포인트 디렉토리의 모든 .pt 파일 찾기
            checkpoint_files = []
            for file in os.listdir(ckpt_dir):
                if file.endswith('.pt') and file != f'{current_iter}.pt':
                    checkpoint_files.append(os.path.join(ckpt_dir, file))

            # 이전 체크포인트 파일들 삭제
            for ckpt_file in checkpoint_files:
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
                    logger.info(f'Removed old checkpoint: {os.path.basename(ckpt_file)}')

        except Exception as e:
            logger.warning(f'Failed to cleanup old checkpoints: {e}')

    try:
        best_loss, best_iter = None, None
        ## 새로 추가 ##
        patience = 100  # 예: 100번의 validation 동안 개선이 없으면 중단
        patience_counter = 0
        ##
        for it in range(start_it, config.train.max_iters):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if config.data.name == 'pdbbind':
                    _ = test(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    patience_counter = 0  # 개선되었으므로 카운터 초기화

                    # 새로운 체크포인트 저장
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)

                    # 이전 체크포인트 파일들 삭제
                    cleanup_old_checkpoints(ckpt_dir, it, logger)

                else:
                    patience_counter += 1 # 개선되지 않았으므로 카운터 증가
                    logger.info(f'[Validate] Val loss is not improved. for {patience_counter} time(s).'
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
                    ## 새로 추가 ##
                    if patience_counter >= patience:
                        logger.info(f'Early stopping at iteration {it}.')
                        break # 루프 중단
    except KeyboardInterrupt:
        logger.info('Terminating...')
        
        
if __name__ == '__main__':
    main()