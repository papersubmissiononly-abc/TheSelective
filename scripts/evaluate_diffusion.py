import argparse
import os
import sys
# 시스템 경로에 현재 디렉토리 추가하여 내부 모듈 임포트
sys.path.append(os.path.abspath('./'))

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

# 평가, 재구성, 변환 등 유틸리티 모듈 임포트
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask

# 원자 친화도 가중치 파싱을 위한 헬퍼 함수
def parse_affinity_weight(weight_str):
    """
    Parse affinity weight from string format to float
    Handles cases like '[0.]', '0.5', etc.
    """
    try:
        # Try direct float conversion first
        return float(weight_str)
    except ValueError:
        # Handle string representations of arrays like '[0.]'
        import ast
        try:
            # Parse as Python literal (list, tuple, etc.)
            parsed = ast.literal_eval(weight_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return float(parsed[0])
            else:
                return 0.0
        except:
            # Fallback: extract numbers using regex
            import re
            numbers = re.findall(r'-?\d+\.?\d*', weight_str)
            if numbers:
                return float(numbers[0])
            else:
                return 0.0

# 딕셔너리 내용을 깔끔하게 출력하기 위한 헬퍼 함수
def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

# 생성된 분자들의 고리(ring) 크기 분포를 출력하기 위한 헬퍼 함수
def print_ring_ratio(all_ring_sizes, logger):
    ring_info = {}
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')
        ring_info[ring_size] = f'{n_mol / len(all_ring_sizes):.3f}'
    return ring_info

def main():
     # --- 1. 초기 설정 및 구성 (Setup and Configuration) ---
    # 커맨드 라인 인자(argument) 파싱
    parser = argparse.ArgumentParser()
    # WARN: important turn on when evaluate pdbbind related proteins
    ################
    parser.add_argument('--eval_pdbbind', action='store_true')
    ################
    
    parser.add_argument('--sample_path', type=str, default='./test_poc/')
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--save_complex', action='store_true', default=False, help='Save protein-ligand complex')
    parser.add_argument('--protein_root', type=str, default='./scratch2/data/test_set/')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()

    # 결과 저장 경로 및 로거 설정
    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')# RDKit의 불필요한 로그 비활성화

    # --- 2. 생성된 데이터 로드 (Load Generated Data) ---
    # sample_path에서 'result_*.pt' 패턴을 가진 모든 파일 검색
    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    # 평가 지표들을 누적하기 위한 변수 초기화
    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    # --- 3. 메인 평가 루프 (Main Evaluation Loop) ---
    # 각 단백질 타겟에 대해 생성된 결과 파일을 하나씩 순회
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        
        # 샘플링 결과(.pt) 파일 로드
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj'] # [샘플 수, 스텝 수, 원자 수, 유형]
        all_pred_exp_traj = r['pred_exp_traj']
        all_pred_exp_score = r['pred_exp']
        all_pred_exp_atom_traj = r['pred_exp_atom_traj']
        # all_pred_exp_atom_traj = [np.zeros_like(all_pred_ligand_v[0]) for i in range(len(all_pred_exp_score))]
        num_samples += len(all_pred_ligand_pos)

        # 생성된 각 샘플(분자)에 대해 순회
        for sample_idx, (pred_pos, pred_v, pred_exp_score, pred_exp_atom_weight) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v, all_pred_exp_score, all_pred_exp_atom_traj)):
            # 평가할 특정 스텝의 데이터 추출 (-1은 마지막 스텝)
            pred_pos, pred_v, pred_exp, pred_exp_atom_weight = pred_pos[args.eval_step], pred_v[args.eval_step], pred_exp_score, pred_exp_atom_weight[args.eval_step]

            # --- 3a. 원자 안정성 평가 (Atom Stability Check) ---
            # 원자 인덱스를 원자 번호로 변환
            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            # 원자가 너무 가깝게 붙어 있는지 확인하여 안정성 평가
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0] # 분자 단위 안정성
            all_atom_stable += r_stable[1] # 원자 단위 안정성
            all_n_atom += r_stable[2] # 전체 원자 수

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # --- 3b. 분자 재구성 (Molecule Reconstruction) ---
            # 논문의 Discussion 섹션에서 언급된 후처리 과정에 해당.
            # 생성된 원자 좌표와 종류로부터 실제 RDKit 분자 객체(mol)를 생성.
            # 이 과정에서 원자 간의 결합(bond)이 형성됨.
            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, pred_exp_atom_weight)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue # 재구성 실패 시 다음 샘플로 넘어감
            n_recon_success += 1
            
            # 재구성된 분자가 여러 조각으로 나뉘었는지 확인
            if '.' in smiles:
                continue # 단일 분자가 아니면 건너뜀
            n_complete += 1
            
            # --- 3c. 화학적 특성 및 도킹 점수 평가 (Chemical Properties and Docking Score) ---
            # 논문의 Table 1에 나온 평가 지표들을 계산하는 부분.
            # chemical and docking check
            # try:
                 # QED(약물 적합성), SA(합성 가능성) 등 화학적 특성 계산
            chem_results = scoring_func.get_chem(mol)

                
            if args.docking_mode == 'qvina':
                vina_task = QVinaDockingTask.from_generated_mol(
                    mol, r['data'].protein_filename, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()
            # Vina를 이용한 도킹 시뮬레이션 수행
            elif args.docking_mode in ['vina_score', 'vina_dock']:
                if args.eval_pdbbind:
                    logger.info('eval pdbbind')
                    # 단백질 파일 경로 설정
                    protein_fn = os.path.join(
                        os.path.dirname(r['data'].ligand_filename),
                        os.path.basename(r['data'].ligand_filename)[:4] + '_protein.pdb'
                    )
                else:
                    logger.info('eval other dataset')
                    # 단백질 파일 경로 설정
                    protein_fn = os.path.join(
                        os.path.dirname(r['data'].ligand_filename),
                        os.path.basename(r['data'].ligand_filename)[:10] + '.pdb'
                    )
                # Vina Score: 생성된 3D 구조 그대로 점수 계산
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, protein_fn, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                # Vina Minimize: 3D 구조를 국소적으로 최적화한 후 점수 계산
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                # Vina Dock: 포켓 내에서 전체적으로 최적의 위치와 구조를 찾은 후 점수 계산
                if args.docking_mode == 'vina_dock':
                    docking_save_path = None
                    if args.save_complex:
                        complex_path = os.path.join(result_path, 'complexes')
                        os.makedirs(complex_path, exist_ok=True)
                        docking_save_path = os.path.join(complex_path, f'{example_idx}_{sample_idx}_dock.pdb')
                    docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness, save_path=docking_save_path)
                    vina_results['dock'] = docking_results
                
                sdf_path = os.path.join(result_path, f"sdf_{r_name[:-3].split('_')[-1]}")
                os.makedirs(sdf_path, exist_ok=True)
                writer = Chem.SDWriter(os.path.join(sdf_path, f'res_{sample_idx}.sdf'))
                writer.write(mol)
                writer.close()
            else:
                vina_results = None

            n_eval_success += 1
            # except:
            #     if args.verbose:
            #         logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
            #     continue # 평가 실패 시 다음 샘플로 넘어감

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            # 성공적으로 평가된 분자들의 정보를 results 리스트에 저장
            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results,
                'pred_exp': pred_exp,
                'atom_exp': {
                    atom.GetIdx(): parse_affinity_weight(atom.GetProp('_affinity_weight')) for atom in mol.GetAtoms()
                }
            })
    logger.info(f'Evaluate done! {num_samples} samples in total.')

# --- 4. 최종 결과 집계 및 출력 (Final Result Aggregation and Logging) ---
    # 유효성(validity), 안정성(stability) 등 계산
    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples # 단일 조각으로 성공적으로 재구성된 비율
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

 # 결합 길이, 원자 종류 분포의 JS Divergence 계산 (생성된 분자 분포와 실제 분포의 유사도 측정)
    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    if np.isnan(atom_type_js):
        logger.info('Atom type JS: NaN (no valid atoms found)')
    else:
        logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    # 주요 화학적 속성 및 도킹 점수의 평균/중앙값 계산 및 출력
    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    ring_info = print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)


    # 생성된 모든 결과와 지표들을 파일로 저장
    if args.save:
        torch.save({
            'info': 'Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)),
            'ring_info': ring_info,
            'stability': validity_dict,
            'c_bond_length_dict': c_bond_length_dict,
            'success_js_metrics': success_js_metrics,
            'atom_type_js': atom_type_js,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}_wo_vina.pt'))


if __name__ == '__main__':
    main()