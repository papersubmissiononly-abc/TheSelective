import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from utils.data import PDBProtein, parse_sdf_file
from rdkit import Chem
from rdkit.Chem import AllChem

# Multi-target functionality imports
import json


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    # Handle empty trajectory case (e.g., when --on_target_only is used)
    if not ligand_v_traj:
        # Return empty arrays with proper structure
        return [np.array([]).reshape(0, 0) for _ in range(n_data)]
    
    for v in ligand_v_traj:  # step_i
        # Detach tensor before converting to numpy to handle gradients
        v_array = v.detach().cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    # Only stack if there are elements to stack
    all_step_v = [np.stack(step_v) if step_v else np.array([]).reshape(0, 0) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

"""
    KGDiff 모델을 사용하여 특정 단백질 포켓에 결합할 리간드 분자를 생성하는 핵심 함수.
    논문의 Denoising Process(그림 1)에 해당하며, 노이즈로부터 분자를 점진적으로 복원합니다.
    Args:
        model (ScorePosNet3D): 훈련된 KGDiff 확산 모델 (논문의 Φθ에 해당).
        data (torch_geometric.data.Data): 단백질 포켓 정보를 담은 데이터 객체.
        num_samples (int): 생성할 분자(샘플)의 총 개수.
        batch_size (int): 한 번에 처리할 샘플의 수.
        device (str): 계산을 수행할 장치 ('cuda' 사용).
        num_steps (int): Denoising 단계 수 (논문의 T).
        center_pos_mode (str): 리간드 초기 위치 설정 모드.
        sample_num_atoms (str): 생성될 리간드의 원자 수를 결정하는 방식 ('prior', 'ref' 등).
        guide_mode (str): 가이던스 방식 ('joint', 'valuenet' 등). 'joint'는 논문에서 제안된 방식.
        value_model: 가이던스를 위한 별도의 전문가 네트워크 (선택 사항).
        type_grad_weight (float): 원자 유형(discrete)에 대한 가이던스 강도 (논문의 r).
        pos_grad_weight (float): 원자 좌표(continuous)에 대한 가이던스 강도 (논문의 s).
    Returns:
        tuple: 생성된 분자의 위치, 원자 종류, 예측된 결합 친화도(affinity), 그리고 각 속성의 전체 생성 궤적(trajectory) 등을 반환.
"""
def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda',
                            num_steps=None, center_pos_mode='protein', center_ligand=False,
                            sample_num_atoms='prior',guide_mode='joint',
                            value_model=None,
                            type_grad_weight=1.,pos_grad_weight=1., w_off=1.0, w_on=1.0,
                            off_target_data=None, off_grad_weight=1.0,
                            off_type_grad_weight=None, off_pos_grad_weight=None, on_target_only=False,
                            head1_type_grad_weight=None, head1_pos_grad_weight=None,
                            head2_type_grad_weight=None, head2_pos_grad_weight=None):

    # Default off-target weights to same as on-target if not specified
    if off_type_grad_weight is None:
        off_type_grad_weight = type_grad_weight
    if off_pos_grad_weight is None:
        off_pos_grad_weight = pos_grad_weight

    # Default head-specific weights to general weights if not specified
    if head1_type_grad_weight is None:
        head1_type_grad_weight = type_grad_weight
    if head1_pos_grad_weight is None:
        head1_pos_grad_weight = pos_grad_weight
    if head2_type_grad_weight is None:
        head2_type_grad_weight = type_grad_weight
    if head2_pos_grad_weight is None:
        head2_pos_grad_weight = pos_grad_weight
    
    # 생성된 분자의 최종 상태와 궤적을 저장할 리스트 초기화
    all_pred_pos, all_pred_v, all_pred_exp = [], [], []
    all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_exp_atom_traj = [], [], [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    all_pred_exp_off, all_pred_exp_off_traj = [], []  # Off-target tracking
    all_pred_exp_on, all_pred_exp_on_traj = [], []    # On-target tracking

    # FIX: For multi-batch trajectory collection, use dictionaries indexed by timestep
    # This allows proper accumulation of samples across batches for each timestep
    all_pred_exp_traj_dict = {}       # {timestep: [batch1_samples, batch2_samples, ...]}
    all_pred_exp_off_traj_dict = {}   # {timestep: [batch1_samples, batch2_samples, ...]}
    all_pred_exp_on_traj_dict = {}    # {timestep: [batch1_samples, batch2_samples, ...]}

    time_list = []
    # 전체 샘플을 배치 크기 단위로 나누어 처리
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        # Create batch following train_diffusion.py pattern
        if on_target_only:
            # ON-TARGET ONLY MODE: Create single protein batch like original KGDiff
            # This is equivalent to original sample_diffusion_original.py joint mode
            batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
            off_target_batches = None
            
        elif guide_mode == 'selectivity' and off_target_data is not None:
            # For selectivity mode: create combined multi-protein batch like in training
            # Handle both list and single object cases
            if isinstance(off_target_data, list):
                # If it's a list, combine the off-targets into a single object
                off_data = combine_off_targets_for_guidance(off_target_data)
            else:
                # Already a single combined object
                off_data = off_target_data
            
            # Create batch from combined multi-protein data
            batch = Batch.from_data_list([off_data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
            off_target_batches = None
        elif guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
            # For sequential selectivity modes: use on-target only for initial batch creation
            # Off-target data will be used sequentially during sampling
            batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

            # Store off-target data for sequential processing during sampling
            # Memory optimization: Limit number of off-targets to reduce memory pressure
            if off_target_data is None:
                if guide_mode in ['pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                    # Allow these modes without off-targets (on-target only)
                    print(f"{guide_mode} mode: Running in on-target only mode (no off-target data)")
                    off_target_data = []  # Empty list for on-target only
                else:
                    raise ValueError(f"off_target_data is required for {guide_mode} mode")
            else:
                # Ensure off_target_data is a list of data objects
                if not isinstance(off_target_data, list):
                    off_target_data = [off_target_data]

                # Memory optimization: Limit number of off-targets processed
                MAX_OFF_TARGETS = 3  # Process maximum 3 off-targets to avoid OOM
                if len(off_target_data) > MAX_OFF_TARGETS:
                    print(f"Warning: Limiting off-targets from {len(off_target_data)} to {MAX_OFF_TARGETS} for memory efficiency")
                    off_target_data = off_target_data[:MAX_OFF_TARGETS]

            # off_target_batches will be None - sequential processing doesn't use batches
            off_target_batches = None

        else:
            # Standard mode: single protein batch
            batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
            off_target_batches = None

        t1 = time.time()
        
        # Memory optimization: Clear cache before processing each batch
        if guide_mode in ['selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential'] and device == 'cuda':
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Use no_grad for all modes (like original), gradients will be enabled inside guidance functions when needed
        # Also use mixed precision scaler for memory efficiency (only on CUDA)
        use_mixed_precision = guide_mode in ['selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential'] and device == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision) if use_mixed_precision else None
        
        # All modes use no_grad() at top level, following original design
        # Guidance functions will enable gradients internally when needed
        with torch.no_grad():
            # Use autocast only on CUDA devices
            if device == 'cuda' and use_mixed_precision:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
            else:
                # Dummy context manager for non-mixed precision
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            
            with autocast_ctx:
                batch_protein = batch.protein_element_batch
                # 생성할 리간드의 원자 수 결정
                if sample_num_atoms == 'prior': # 포켓 크기에 기반한 사전 분포
                    if on_target_only:
                        # ON-TARGET ONLY MODE: Use entire protein pocket (like original KGDiff)
                        pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                        ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                        batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                    else:
                        # Multi-protein mode: On-target pocket만 사용하여 원자 수 결정
                        on_target_mask = batch.protein_id == 0 if hasattr(batch, 'protein_id') else torch.ones_like(batch.protein_element_batch, dtype=torch.bool)
                        on_target_pos = batch.protein_pos[on_target_mask]
                        pocket_size = atom_num.get_space_size(on_target_pos.detach().cpu().numpy())
                        ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                        batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                elif sample_num_atoms == 'range':
                    ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                    batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                elif sample_num_atoms == 'ref':  # 참조(Reference) 리간드의 원자 수 사용
                    batch_ligand = batch.ligand_element_batch
                    ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
                else:
                    raise ValueError

                # Denoising Process의 시작점(T_step)인 노이즈 상태의 리간드 초기화
                # 1. 리간드 원자 위치 초기화 (center_ligand 모드에 따라 결정)
                if on_target_only:
                    # ON-TARGET ONLY MODE: Single protein initialization (like original KGDiff)
                    if center_ligand and hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                        center_pos = scatter_mean(batch.ligand_pos, batch.ligand_element_batch, dim=0)
                    else:
                        center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                        
                elif hasattr(batch, 'protein_id'):
                    # Multi-protein case: center_ligand 모드에 따라 초기화 방식 결정
                    if center_ligand:
                        # Center_ligand=True: 리간드 중심으로 초기화
                        # Training에서와 동일하게 리간드 중심을 기준으로 off-target들이 평행이동됨
                        # 여기서는 원본 리간드가 없으므로, 데이터의 ligand_pos 정보를 사용
                        if hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                            # Use the reference ligand position if available
                            reference_ligand_pos = batch.ligand_pos
                            reference_ligand_batch = batch.ligand_element_batch
                            center_pos = scatter_mean(reference_ligand_pos, reference_ligand_batch, dim=0)
                        else:
                            # Fallback to on-target protein center if no reference ligand
                            on_target_mask = batch.protein_id == 0
                            on_target_pos = batch.protein_pos[on_target_mask]
                            on_target_batch = batch.protein_element_batch[on_target_mask]
                            center_pos = scatter_mean(on_target_pos, on_target_batch, dim=0)
                    else:
                        # Center_ligand=False: 단백질 중심으로 초기화 (기존 방식)
                        on_target_mask = batch.protein_id == 0
                        on_target_pos = batch.protein_pos[on_target_mask]
                        on_target_batch = batch.protein_element_batch[on_target_mask]
                        center_pos = scatter_mean(on_target_pos, on_target_batch, dim=0)
                else:
                    # Single protein case: center_ligand 모드에 따라 초기화
                    if center_ligand and hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                        center_pos = scatter_mean(batch.ligand_pos, batch.ligand_element_batch, dim=0)
                    else:
                        center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                
                # Handle batch indexing for center_pos
                if len(center_pos) == 0:
                    # Fallback: use protein center if center_pos is empty
                    center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                
                # Ensure center_pos has correct size for batch_ligand indexing
                max_batch_idx = batch_ligand.max().item() if len(batch_ligand) > 0 else 0
                if len(center_pos) <= max_batch_idx:
                    # Expand center_pos to match batch size if needed
                    if len(center_pos) == 1:
                        center_pos = center_pos.repeat(max_batch_idx + 1, 1)
                    else:
                        # Use first center position for all batches as fallback
                        center_pos = center_pos[:1].repeat(max_batch_idx + 1, 1)
                
                batch_center_pos = center_pos[batch_ligand]
                init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

                # 2. 리간드 원자 유형 초기화
                # 모든 원자 유형에 대해 균일한 확률 분포로 초기화
                # init ligand v
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v_prob = log_sample_categorical(uniform_logits)
                init_ligand_v = init_ligand_v_prob.argmax(dim=-1)

                # 확산 모델의 샘플링 함수 호출하여 분자 생성
                # 이 함수 내부에서 T_step부터 1_step까지 반복적인 Denoising 수행
                sample_args = {
                    'guide_mode': guide_mode,
                    'value_model': value_model,
                    'type_grad_weight': type_grad_weight,  # 가중치 r
                    'pos_grad_weight': pos_grad_weight,  # 가중치 s
                    'head1_type_grad_weight': head1_type_grad_weight,  # head1 전용 type grad weight
                    'head1_pos_grad_weight': head1_pos_grad_weight,    # head1 전용 pos grad weight
                    'head2_type_grad_weight': head2_type_grad_weight,  # head2 전용 type grad weight
                    'head2_pos_grad_weight': head2_pos_grad_weight,    # head2 전용 pos grad weight
                    'protein_pos': batch.protein_pos,
                    'protein_v': batch.protein_atom_feature.float(),
                    'batch_protein': batch_protein,
                    'init_ligand_pos': init_ligand_pos,
                    'init_ligand_v': init_ligand_v,
                    'batch_ligand': batch_ligand,
                    'num_steps': num_steps,
                    'center_pos_mode': center_pos_mode,
                    'w_off': w_off,
                    'w_on': w_on,
                    'on_target_only': on_target_only
                }
                
                # Add sequential selectivity specific parameters
                if guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                    # For memory efficiency, only pass one off-target at a time
                    # The model will process them sequentially during sampling
                    # For now, pass the list but the model should process them one by one
                    sample_args['off_target_data'] = off_target_data
                    print(f"[DEBUG sample_diffusion.py] Passing off_target_data to model: {len(off_target_data) if off_target_data else 0} off-targets")

                # Only add protein_id for multi-protein modes
                if not on_target_only:
                    sample_args['protein_id'] = getattr(batch, 'protein_id', None)
                
                # For selectivity mode, off-target data is already included in the combined batch
                # No need for separate off_target_batches parameter
                
                # Forward pass (mixed precision is already handled by autocast_ctx above)
                r = model.sample_diffusion(**sample_args)

            # 생성 결과 (최종 분자 및 전체 궤적) 추출
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] # 예측된 결합 친화도(V) 궤적
            exp_atom_traj = r.get('exp_atom_traj', [])  # Handle missing exp_atom_traj

            # unbatch exp
            # 'joint' 모드일 때, 예측된 결합 친화도(V) 처리
            if guide_mode in ['joint', 'pdbbind_random', 'valuenet', 'wo', 'selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                # 최종 예측된 결합 친화도(exp) 저장
                if len(exp_traj) > 0:
                    all_pred_exp += exp_traj[-1]
                    # FIX: Collect trajectories per timestep for proper multi-batch handling
                    for step_idx, step_exp in enumerate(exp_traj):
                        if step_idx not in all_pred_exp_traj_dict:
                            all_pred_exp_traj_dict[step_idx] = []
                        all_pred_exp_traj_dict[step_idx].append(step_exp[:n_data])

                # Handle off-target predictions for selectivity modes
                if guide_mode in ['selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                    if 'exp_off_traj' in r and len(r['exp_off_traj']) > 0:
                        all_pred_exp_off += r['exp_off_traj'][-1]
                        # FIX: Collect off-target trajectories per timestep
                        for step_idx, step_exp_off in enumerate(r['exp_off_traj']):
                            if step_idx not in all_pred_exp_off_traj_dict:
                                all_pred_exp_off_traj_dict[step_idx] = []
                            all_pred_exp_off_traj_dict[step_idx].append(step_exp_off[:n_data])

                    # Handle on-target predictions
                    if 'exp_on_traj' in r and len(r['exp_on_traj']) > 0:
                        # FIX: Ensure tensor is at least 1D before processing
                        last_exp_on = r['exp_on_traj'][-1]
                        if last_exp_on.dim() == 0:
                            last_exp_on = last_exp_on.unsqueeze(0)
                        # Add each element as 1D tensor to ensure .size(0) works
                        for i in range(last_exp_on.shape[0]):
                            element = last_exp_on[i]
                            # Ensure element is at least 1D
                            if element.dim() == 0:
                                element = element.unsqueeze(0)
                            all_pred_exp_on.append(element)

                        # FIX: Collect on-target trajectories per timestep
                        for step_idx, step_exp_on in enumerate(r['exp_on_traj']):
                            if step_idx not in all_pred_exp_on_traj_dict:
                                all_pred_exp_on_traj_dict[step_idx] = []
                            # FIX: Ensure tensor is 1D before slicing
                            if step_exp_on.dim() == 0:
                                step_exp_on = step_exp_on.unsqueeze(0)
                            all_pred_exp_on_traj_dict[step_idx].append(step_exp_on[:n_data])
            
            # unbatch pos
            # 배치 처리된 결과들을 개별 샘플로 분리 (Unbatching)
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)

            # 1. 최종 원자 위치 분리
            ligand_pos_array = ligand_pos.detach().cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

             
            # 2. 원자 위치 궤적 분리
            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.detach().cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            # 3. 최종 원자 유형 분리
            ligand_v_array = ligand_v.detach().cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

             # 4. 원자 유형 궤적 분리 (헬퍼 함수 사용)
            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
            all_step_exp_atom = unbatch_v_traj(exp_atom_traj, n_data, ligand_cum_atoms)
            all_pred_exp_atom_traj += [v for v in all_step_exp_atom]
            
            
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
        
        # Memory optimization: Clear intermediate tensors after each batch
        if guide_mode == 'selectivity':
            del batch
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        
    # Process on-target predictions
    if all_pred_exp:
        # FIX: Handle tensors with different sizes in multi-protein mode
        # Pad all tensors to the maximum size before stacking
        if len(all_pred_exp) > 0 and torch.is_tensor(all_pred_exp[0]):
            max_size = max(t.size(0) for t in all_pred_exp)
            padded_tensors = []
            for t in all_pred_exp:
                if t.size(0) < max_size:
                    # Pad with -inf (or a sentinel value) to indicate missing values
                    padding = torch.full((max_size - t.size(0),), float('-inf'),
                                        dtype=t.dtype, device=t.device)
                    t = torch.cat([t, padding], dim=0)
                padded_tensors.append(t)
            all_pred_exp = torch.stack(padded_tensors, dim=0).numpy()
        else:
            all_pred_exp = torch.stack(all_pred_exp, dim=0).numpy()
    else:
        all_pred_exp = np.array([])

    # FIX: Convert trajectory dictionaries to final arrays
    # For each timestep, concatenate samples from all batches
    if all_pred_exp_traj_dict:
        all_pred_exp_traj = []
        for step_idx in sorted(all_pred_exp_traj_dict.keys()):
            # FIX: Handle tensors with different sizes in multi-protein mode
            # Pad all tensors to the same size before concatenating
            batch_tensors = all_pred_exp_traj_dict[step_idx]
            if len(batch_tensors) > 0 and batch_tensors[0].dim() > 1:
                # Multi-protein mode: tensors have shape [batch_size, num_proteins]
                max_proteins = max(t.size(-1) for t in batch_tensors)
                padded_batches = []
                for t in batch_tensors:
                    if t.size(-1) < max_proteins:
                        # Pad along the last dimension (proteins)
                        pad_size = max_proteins - t.size(-1)
                        padding = torch.full((*t.shape[:-1], pad_size), float('-inf'),
                                           dtype=t.dtype, device=t.device)
                        t = torch.cat([t, padding], dim=-1)
                    padded_batches.append(t)
                timestep_samples = torch.cat(padded_batches, dim=0)
            else:
                # Single-protein mode: can concatenate directly
                timestep_samples = torch.cat(batch_tensors, dim=0)
            all_pred_exp_traj.append(timestep_samples)
        # Stack all timesteps: [num_steps, num_samples, num_proteins]
        all_pred_exp_traj = torch.stack(all_pred_exp_traj, dim=0).numpy()
    else:
        all_pred_exp_traj = np.array([])

    # Process off-target predictions
    if all_pred_exp_off:
        # FIX: Handle tensors with different sizes in multi-protein mode
        if len(all_pred_exp_off) > 0 and torch.is_tensor(all_pred_exp_off[0]):
            max_size = max(t.size(0) for t in all_pred_exp_off)
            padded_tensors = []
            for t in all_pred_exp_off:
                if t.size(0) < max_size:
                    padding = torch.full((max_size - t.size(0),), float('-inf'),
                                        dtype=t.dtype, device=t.device)
                    t = torch.cat([t, padding], dim=0)
                padded_tensors.append(t)
            all_pred_exp_off = torch.stack(padded_tensors, dim=0).numpy()
        else:
            all_pred_exp_off = torch.stack(all_pred_exp_off, dim=0).numpy()
    else:
        all_pred_exp_off = np.array([])

    # FIX: Convert off-target trajectory dictionary to final array
    if all_pred_exp_off_traj_dict:
        all_pred_exp_off_traj = []
        for step_idx in sorted(all_pred_exp_off_traj_dict.keys()):
            # FIX: Handle tensors with different sizes in multi-protein mode
            batch_tensors = all_pred_exp_off_traj_dict[step_idx]
            if len(batch_tensors) > 0 and batch_tensors[0].dim() > 1:
                max_proteins = max(t.size(-1) for t in batch_tensors)
                padded_batches = []
                for t in batch_tensors:
                    if t.size(-1) < max_proteins:
                        pad_size = max_proteins - t.size(-1)
                        padding = torch.full((*t.shape[:-1], pad_size), float('-inf'),
                                           dtype=t.dtype, device=t.device)
                        t = torch.cat([t, padding], dim=-1)
                    padded_batches.append(t)
                timestep_samples = torch.cat(padded_batches, dim=0)
            else:
                timestep_samples = torch.cat(batch_tensors, dim=0)
            all_pred_exp_off_traj.append(timestep_samples)
        # Stack all timesteps: [num_steps, num_samples, num_proteins]
        all_pred_exp_off_traj = torch.stack(all_pred_exp_off_traj, dim=0).numpy()
    else:
        all_pred_exp_off_traj = np.array([])

    # Process on-target predictions (separate from off-target)
    if all_pred_exp_on:
        # FIX: Handle tensors with different sizes in multi-protein mode
        if len(all_pred_exp_on) > 0 and torch.is_tensor(all_pred_exp_on[0]):
            max_size = max(t.size(0) for t in all_pred_exp_on)
            padded_tensors = []
            for t in all_pred_exp_on:
                if t.size(0) < max_size:
                    padding = torch.full((max_size - t.size(0),), float('-inf'),
                                        dtype=t.dtype, device=t.device)
                    t = torch.cat([t, padding], dim=0)
                padded_tensors.append(t)
            all_pred_exp_on = torch.stack(padded_tensors, dim=0).numpy()
        else:
            all_pred_exp_on = torch.stack(all_pred_exp_on, dim=0).numpy()
    else:
        all_pred_exp_on = np.array([])

    # FIX: Convert on-target trajectory dictionary to final array
    if all_pred_exp_on_traj_dict:
        all_pred_exp_on_traj = []
        for step_idx in sorted(all_pred_exp_on_traj_dict.keys()):
            # FIX: Handle tensors with different sizes in multi-protein mode
            batch_tensors = all_pred_exp_on_traj_dict[step_idx]
            if len(batch_tensors) > 0 and batch_tensors[0].dim() > 1:
                max_proteins = max(t.size(-1) for t in batch_tensors)
                padded_batches = []
                for t in batch_tensors:
                    if t.size(-1) < max_proteins:
                        pad_size = max_proteins - t.size(-1)
                        padding = torch.full((*t.shape[:-1], pad_size), float('-inf'),
                                           dtype=t.dtype, device=t.device)
                        t = torch.cat([t, padding], dim=-1)
                    padded_batches.append(t)
                timestep_samples = torch.cat(padded_batches, dim=0)
            else:
                timestep_samples = torch.cat(batch_tensors, dim=0)
            all_pred_exp_on_traj.append(timestep_samples)
        # Stack all timesteps: [num_steps, num_samples, num_proteins]
        all_pred_exp_on_traj = torch.stack(all_pred_exp_on_traj, dim=0).numpy()
    else:
        all_pred_exp_on_traj = np.array([])

    return (all_pred_pos, all_pred_v, all_pred_exp, all_pred_pos_traj, all_pred_v_traj,
            all_pred_exp_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_exp_atom_traj,
            all_pred_exp_off, all_pred_exp_off_traj, all_pred_exp_on, all_pred_exp_on_traj, time_list)

def load_custom_test_set(test_set_path, transform=None):
    """
    Load custom test set and create ProteinLigandData objects that mimic MultiProteinPairedDataset format
    
    Args:
        test_set_path (str): Path to test set directory (e.g., data/multipro_validation_test_set)
        transform: Data transformation to apply
        
    Returns:
        list: List of ProteinLigandData objects compatible with MultiProteinPairedDataset
    """
    import glob
    import torch

    test_data = []
    protein_dirs = glob.glob(os.path.join(test_set_path, "*"))
    protein_dirs = [d for d in protein_dirs if os.path.isdir(d)]
    
    for protein_dir in sorted(protein_dirs):
        protein_name = os.path.basename(protein_dir)
        
        # Find protein and ligand files
        protein_files = glob.glob(os.path.join(protein_dir, "*_rec.pdb"))
        ligand_files = glob.glob(os.path.join(protein_dir, "*.sdf"))
        
        for protein_file in protein_files:
            # Find corresponding ligand files for this protein
            protein_base = os.path.basename(protein_file).replace('.pdb', '')
            # Ligand files start with the protein_base
            corresponding_ligands = [lf for lf in ligand_files if os.path.basename(lf).startswith(protein_base + '_')]
            
            for ligand_file in corresponding_ligands:
                try:
                    # Load protein and ligand
                    protein_dict = PDBProtein(protein_file).to_dict_atom()
                    ligand_dict = parse_sdf_file(ligand_file)
                    
                    # Prepare protein data as combined (mimicking MultiProteinPairedDataset)
                    # Since we only have on-target protein, protein_id will be all zeros
                    protein_id = torch.zeros(len(protein_dict['element']), dtype=torch.long)
                    
                    # Handle hybridization list of strings properly
                    hybridization = ligand_dict.get('hybridization', [])
                    if hybridization and isinstance(hybridization, list) and isinstance(hybridization[0], str):
                        # Convert string hybridization to numerical values
                        hyb_map = {'SP': 0, 'SP2': 1, 'SP3': 2, 'PLANAR3': 3, 'TETRAHEDRAL': 4, 'TRIGONAL': 5}
                        hybridization_numerical = [hyb_map.get(h, 0) for h in hybridization]
                    else:
                        hybridization_numerical = hybridization if hybridization else [0] * len(ligand_dict['element'])
                    
                    # Create ProteinLigandData object similar to MultiProteinPairedDataset output
                    data_dict = {
                        # Combined protein attributes (mimicking MultiProteinPairedDataset.__getitem__)
                        'protein_pos': torch.tensor(protein_dict['pos'], dtype=torch.float),
                        'protein_element': torch.tensor(protein_dict['element'], dtype=torch.long),
                        'protein_is_backbone': torch.tensor(protein_dict['is_backbone'], dtype=torch.bool),
                        'protein_atom_to_aa_type': torch.tensor(protein_dict['atom_to_aa_type'], dtype=torch.long),
                        'protein_id': protein_id,
                        
                        # Ligand attributes
                        'ligand_pos': torch.tensor(ligand_dict['pos'], dtype=torch.float),
                        'ligand_element': torch.tensor(ligand_dict['element'], dtype=torch.long),
                        'ligand_bond_index': torch.tensor(ligand_dict['bond_index'], dtype=torch.long).t().contiguous() if 'bond_index' in ligand_dict and len(ligand_dict['bond_index']) > 0 else torch.empty((2, 0), dtype=torch.long),
                        'ligand_bond_type': torch.tensor(ligand_dict['bond_type'], dtype=torch.long) if 'bond_type' in ligand_dict and len(ligand_dict['bond_type']) > 0 else torch.empty(0, dtype=torch.long),
                        'ligand_atom_feature': torch.tensor(ligand_dict['atom_feature'], dtype=torch.float) if 'atom_feature' in ligand_dict else torch.zeros(len(ligand_dict['element']), 8, dtype=torch.float),
                        'ligand_hybridization': torch.tensor(hybridization_numerical, dtype=torch.long),
                        
                        # Affinities - set to default values for single protein case
                        'on_target_affinity': torch.tensor(5.0, dtype=torch.float32),  # Default binding affinity
                        'off_target_affinities': torch.empty(0, dtype=torch.float32),  # Empty off-target affinities
                        'affinity': torch.tensor(5.0, dtype=torch.float32),  # For transforms
                        'id': len(test_data)
                    }
                    
                    # Create ProteinLigandData object
                    data = ProteinLigandData(**data_dict)
                    
                    # Add custom attributes for reference
                    data.protein_name = protein_name
                    data.protein_file = protein_file
                    data.ligand_file = ligand_file
                    
                    # Apply transform if provided
                    if transform is not None:
                        data = transform(data)
                    
                    test_data.append(data)
                    
                except Exception as e:
                    print(f"Error loading {protein_file} and {ligand_file}: {e}")
                    continue
    
    return test_data

def extract_protein_pocket(protein_dict, ligand_center_pos, pocket_radius=10.0):
    """
    Extract protein pocket atoms within radius from ligand center
    
    Args:
        protein_dict (dict): Dictionary containing protein information
        ligand_center_pos (torch.Tensor): Center position of ligand [3]
        pocket_radius (float): Radius to extract pocket atoms
        
    Returns:
        dict: Dictionary containing pocket protein information
    """
    protein_pos = protein_dict['pos']
    
    # Convert to torch tensors for consistent computation
    if isinstance(protein_pos, np.ndarray):
        protein_pos_tensor = torch.from_numpy(protein_pos).float()
    else:
        protein_pos_tensor = protein_pos.float()
    
    if isinstance(ligand_center_pos, np.ndarray):
        ligand_center_tensor = torch.from_numpy(ligand_center_pos).float()
    else:
        ligand_center_tensor = ligand_center_pos.float()
    
    # Calculate distances from ligand center
    distances = torch.norm(protein_pos_tensor - ligand_center_tensor, dim=1)
    pocket_mask = distances <= pocket_radius
    
    # Convert mask back to numpy for indexing
    pocket_mask_np = pocket_mask.numpy().astype(bool)
    
    # Extract pocket atoms
    pocket_dict = {}
    for key, value in protein_dict.items():
        if isinstance(value, (list, np.ndarray)) and len(value) == len(protein_pos):
            if isinstance(value, list):
                pocket_dict[key] = [value[i] for i in range(len(value)) if pocket_mask_np[i]]
            else:
                pocket_dict[key] = value[pocket_mask_np]
        else:
            pocket_dict[key] = value  # Keep non-atom-level attributes
    
    return pocket_dict

def extract_pdb_id_from_path(path):
    """
    Extract PDB ID from off-target path
    
    Args:
        path (str): Path like 'data/multipro_validation_test_set/ALDH2_HUMAN_18_517_0'
        
    Returns:
        str: PDB ID like '2vle_A' or None if not found
    """
    import glob
    
    if not os.path.exists(path):
        return None
        
    # Find .pdb files in the directory
    pdb_files = glob.glob(os.path.join(path, "*_rec.pdb"))
    
    if pdb_files:
        pdb_file = pdb_files[0]  # Take the first one
        basename = os.path.basename(pdb_file)
        # Extract PDB ID from filename like '2vle_A_rec.pdb' -> '2vle_A'
        pdb_id = basename.replace('_rec.pdb', '')
        return pdb_id
    
    return None

def load_off_target_from_lmdb_random(lmdb_path, on_target_data_id, num_off_targets=3, transform=None):
    """
    Load off-target protein data from LMDB by randomly selecting entries (excluding on-target)
    
    Args:
        lmdb_path (str): Path to LMDB database
        on_target_data_id (int): Data ID of on-target to exclude
        num_off_targets (int): Number of off-target entries to load
        transform: Data transformation to apply
        
    Returns:
        list: List of ProteinLigandData objects for off-targets
    """
    
    import lmdb
    import pickle
    
    # Extract PDB IDs from paths
    target_pdb_ids = []
    for path in off_target_paths:
        pdb_id = extract_pdb_id_from_path(path)
        if pdb_id:
            target_pdb_ids.append((pdb_id, os.path.basename(path)))
            print(f"Target PDB ID: {pdb_id} from {os.path.basename(path)}")
    
    if not target_pdb_ids:
        print("No valid PDB IDs found in off-target paths")
        return None
    
    off_target_data = []
    found_pdb_ids = set()  # Track found PDB IDs to avoid duplicates
    
    # Open LMDB database
    db = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with db.begin() as txn:
            cursor = txn.cursor()
            
            # Search through LMDB entries for matching PDB IDs
            for key, value in cursor:
                try:
                    data = pickle.loads(value)
                    
                    # Check if any off-target proteins match our target PDB IDs
                    if 'off_target_proteins' in data and data['off_target_proteins']:
                        off_target_filenames = data.get('off_target_protein_filenames', [])
                        
                        for i, off_target_protein in enumerate(data['off_target_proteins']):
                            if i < len(off_target_filenames):
                                protein_filename = off_target_filenames[i]
                                
                                if protein_filename:
                                    # Extract PDB ID from filename (e.g., '2vle_A_rec_...pdb' -> '2vle_A')
                                    basename = os.path.basename(protein_filename)
                                    # Handle different filename patterns
                                    for pdb_id, protein_name in target_pdb_ids:
                                        if pdb_id in basename and pdb_id not in found_pdb_ids:
                                            try:
                                                # Create ProteinLigandData for this off-target
                                                data_dict = {
                                                    'protein_pos': torch.tensor(off_target_protein['protein_pos'], dtype=torch.float),
                                                    'protein_element': torch.tensor(off_target_protein['protein_element'], dtype=torch.long),
                                                    'protein_is_backbone': torch.tensor(off_target_protein['protein_is_backbone'], dtype=torch.bool),
                                                    'protein_atom_to_aa_type': torch.tensor(off_target_protein['protein_atom_to_aa_type'], dtype=torch.long),
                                                    'protein_id': torch.ones(len(off_target_protein['protein_element']), dtype=torch.long) * (len(off_target_data) + 1),
                                                    
                                                    # Dummy ligand data (not used in off-target evaluation)
                                                    'ligand_pos': torch.empty(0, 3, dtype=torch.float),
                                                    'ligand_element': torch.empty(0, dtype=torch.long),
                                                    'ligand_bond_index': torch.empty(2, 0, dtype=torch.long),
                                                    'ligand_bond_type': torch.empty(0, dtype=torch.long),
                                                    'ligand_atom_feature': torch.empty(0, 8, dtype=torch.float),
                                                    'ligand_hybridization': torch.empty(0, dtype=torch.long),
                                                    
                                                    # Affinity values
                                                    'on_target_affinity': torch.tensor(0.0, dtype=torch.float32),
                                                    'off_target_affinities': torch.empty(0, dtype=torch.float32),
                                                    'affinity': torch.tensor(0.0, dtype=torch.float32),
                                                    'id': len(off_target_data)
                                                }
                                                
                                                off_data = ProteinLigandData(**data_dict)
                                                off_data.protein_name = protein_name
                                                off_data.protein_filename = protein_filename
                                                off_data.pdb_id = pdb_id
                                                
                                                # Apply transform if provided
                                                if transform is not None:
                                                    off_data = transform(off_data)
                                                
                                                off_target_data.append(off_data)
                                                found_pdb_ids.add(pdb_id)  # Mark as found
                                                print(f"Loaded off-target from LMDB: {protein_name} (PDB: {pdb_id}) with {len(off_target_protein['protein_element'])} atoms")
                                                break
                                                
                                            except Exception as e:
                                                print(f"Error creating off-target data for {pdb_id}: {e}")
                                                continue
                    
                    # Stop if we found all target proteins
                    if len(off_target_data) >= len(target_pdb_ids):
                        break
                        
                except Exception as e:
                    continue
                    
    finally:
        db.close()
    
    return off_target_data if off_target_data else None

def process_ligand_hybridization(hybridization_data):
    """
    Process ligand hybridization data to tensor format
    """
    if isinstance(hybridization_data, list):
        # Handle list of strings
        if hybridization_data and isinstance(hybridization_data[0], str):
            hyb_map = {'SP': 0, 'SP2': 1, 'SP3': 2, 'PLANAR3': 3, 'TETRAHEDRAL': 4, 'TRIGONAL': 5}
            return torch.tensor([hyb_map.get(h, 0) for h in hybridization_data], dtype=torch.long)
        else:
            # List of numbers
            return torch.tensor(hybridization_data, dtype=torch.long)
    elif torch.is_tensor(hybridization_data):
        return hybridization_data.clone()
    else:
        # Default case
        return torch.tensor([0], dtype=torch.long)

def load_combined_off_targets_from_lmdb(lmdb_path, on_target_data_id, num_off_targets=3, transform=None):
    """
    Load off-target protein data from LMDB by randomly selecting from multipro_validation_test_set
    and combine them into a single graph like in training.
    (Modified to use validation info instead of entire LMDB for consistency with ID mode)
    
    Args:
        lmdb_path (str): Path to LMDB database
        on_target_data_id (int): Data ID of on-target to exclude
        num_off_targets (int): Number of off-target entries to load
        transform: Data transformation to apply
        
    Returns:
        ProteinLigandData: Single combined off-target data object
    """
    import lmdb
    import pickle
    import random
    
    # Open LMDB database
    db = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with db.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            
            if len(keys) <= num_off_targets:
                print(f"Warning: Not enough entries in LMDB ({len(keys)} total) for {num_off_targets} off-targets")
                return None
            
            # Load validation info to select from multipro_validation_test_set instead of entire LMDB
            validation_info = load_validation_info()
            if not validation_info:
                print("Warning: Validation info not found, falling back to entire LMDB random selection")
                # Fallback to original method
                available_keys = [k for i, k in enumerate(keys) if i != on_target_data_id]
                selected_keys = random.sample(available_keys, min(num_off_targets, len(available_keys)))
            else:
                # Find on-target validation_id to exclude it
                on_target_validation_id = None
                for i, entry in enumerate(validation_info):
                    if entry['idx'] == on_target_data_id:
                        on_target_validation_id = i
                        break
                
                # Get available validation IDs (exclude on-target)
                available_validation_ids = []
                for i, entry in enumerate(validation_info):
                    if i != on_target_validation_id:  # Exclude on-target
                        available_validation_ids.append(i)
                
                if len(available_validation_ids) < num_off_targets:
                    print(f"Warning: Not enough validation entries ({len(available_validation_ids)} available) for {num_off_targets} off-targets")
                    num_off_targets = min(num_off_targets, len(available_validation_ids))
                
                if num_off_targets == 0:
                    print("Error: No off-target validation entries available")
                    return None
                
                # Randomly select off-target validation IDs from multipro_validation_test_set
                selected_validation_ids = random.sample(available_validation_ids, num_off_targets)
                print(f"Randomly selected off-target validation_ids from multipro_validation_test_set: {selected_validation_ids}")
                
                # Convert to data_ids and get keys
                selected_keys = []
                for val_id in selected_validation_ids:
                    entry = validation_info[val_id]
                    data_id = entry['idx']
                    protein_name = entry['protein_dir']
                    if data_id < len(keys):
                        selected_keys.append(keys[data_id])
                        print(f"  Random off-target validation_id {val_id}: {protein_name} (data_id: {data_id})")
                    else:
                        print(f"Warning: data_id {data_id} out of range")
                
                if not selected_keys:
                    print("Error: No valid off-target keys found, falling back to entire LMDB")
                    available_keys = [k for i, k in enumerate(keys) if i != on_target_data_id]
                    selected_keys = random.sample(available_keys, min(num_off_targets, len(available_keys)))
            
            # Get reference ligand data from on-target
            on_target_key = keys[on_target_data_id] if on_target_data_id < len(keys) else None
            on_target_data = pickle.loads(txn.get(on_target_key))
            
            # Collect off-target proteins
            off_target_proteins = []
            
            for i, key in enumerate(selected_keys):
                try:
                    data = pickle.loads(txn.get(key))
                    off_target_protein = data['on_target_protein']
                    off_target_proteins.append(off_target_protein)
                    print(f"Loaded off-target {i+1}: {len(off_target_protein['protein_element'])} protein atoms")
                except Exception as e:
                    print(f"Error loading off-target {i}: {e}")
                    continue
            
            if not off_target_proteins:
                return None
            
            # Combine all off-target proteins into a single graph (like MultiProteinPairedDataset)
            combined_protein_pos = off_target_proteins[0]['protein_pos'].clone()
            combined_protein_element = off_target_proteins[0]['protein_element'].clone()  
            combined_protein_is_backbone = off_target_proteins[0]['protein_is_backbone'].clone()
            combined_protein_atom_to_aa_type = off_target_proteins[0]['protein_atom_to_aa_type'].clone()
            combined_protein_id = torch.ones_like(off_target_proteins[0]['protein_element'], dtype=torch.long)  # First off-target = ID 1
            
            # Append other off-target proteins
            for i, protein in enumerate(off_target_proteins[1:], 2):  # Start from ID 2
                combined_protein_pos = torch.cat([combined_protein_pos, protein['protein_pos']], dim=0)
                combined_protein_element = torch.cat([combined_protein_element, protein['protein_element']], dim=0)
                combined_protein_is_backbone = torch.cat([combined_protein_is_backbone, protein['protein_is_backbone']], dim=0)  
                combined_protein_atom_to_aa_type = torch.cat([combined_protein_atom_to_aa_type, protein['protein_atom_to_aa_type']], dim=0)
                protein_id_tensor = torch.full_like(protein['protein_element'], fill_value=i, dtype=torch.long)
                combined_protein_id = torch.cat([combined_protein_id, protein_id_tensor], dim=0)
            
            # Create combined off-target data object using reference ligand
            data_dict = {
                # Combined protein attributes
                'protein_pos': combined_protein_pos,
                'protein_element': combined_protein_element,
                'protein_is_backbone': combined_protein_is_backbone,
                'protein_atom_to_aa_type': combined_protein_atom_to_aa_type,
                'protein_id': combined_protein_id,
                
                # Use identical ligand data from on-target for perfect compatibility
                'ligand_pos': on_target_data['ligand_pos'].clone(),
                'ligand_element': on_target_data['ligand_element'].clone(),
                'ligand_bond_index': on_target_data['ligand_bond_index'].clone(),
                'ligand_bond_type': on_target_data['ligand_bond_type'].clone(),
                'ligand_atom_feature': on_target_data['ligand_atom_feature'].clone(),
                'ligand_hybridization': process_ligand_hybridization(on_target_data['ligand_hybridization']),
                
                # Affinity values
                'affinity': torch.tensor(0.0, dtype=torch.float32),
                'id': 999  # Special ID for combined off-target
            }
            
            combined_off_target = ProteinLigandData(**data_dict)
            
            # Add metadata
            combined_off_target.protein_filename = f"combined_off_targets_{num_off_targets}"
            combined_off_target.ligand_filename = "reference_ligand"
            
            # Apply transform if provided
            if transform is not None:
                combined_off_target = transform(combined_off_target)
            
            print(f"Created combined off-target graph: {len(combined_protein_element)} total protein atoms")
            return [combined_off_target]  # Return as list for compatibility
            
    finally:
        db.close()
    
    return None

def load_random_off_targets_from_lmdb(lmdb_path, on_target_data_id, num_off_targets=3, transform=None):
    """
    Load off-target protein data from LMDB by randomly selecting entries (excluding on-target)
    NOTE: This function is deprecated - use load_combined_off_targets_from_lmdb instead
    which now selects from multipro_validation_test_set for consistency
    
    Args:
        lmdb_path (str): Path to LMDB database
        on_target_data_id (int): Data ID of on-target to exclude
        num_off_targets (int): Number of off-target entries to load
        transform: Data transformation to apply
        
    Returns:
        list: List of ProteinLigandData objects for off-targets
    """
    import lmdb
    import pickle
    import random
    
    # Open LMDB database
    db = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with db.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            
            if len(keys) <= num_off_targets:
                print(f"Warning: Not enough entries in LMDB ({len(keys)} total) for {num_off_targets} off-targets")
                return None
            
            # Exclude on-target key and randomly select off-target keys
            on_target_key = keys[on_target_data_id] if on_target_data_id < len(keys) else None
            available_keys = [k for i, k in enumerate(keys) if i != on_target_data_id]
            
            # Randomly select off-target keys
            selected_keys = random.sample(available_keys, min(num_off_targets, len(available_keys)))
            
            off_target_data = []
            
            for i, key in enumerate(selected_keys):
                try:
                    data = pickle.loads(txn.get(key))
                    
                    # Extract off-target protein pocket data (same structure as on-target)
                    off_target_protein = data['on_target_protein']
                    
                    # Use identical ligand data from on-target for tensor compatibility
                    # This ensures perfect batch compatibility between on-target and off-target
                    # (off-target will just ignore the ligand during processing)
                    
                    data_dict = {
                        # Protein attributes (pocket only) - same as on-target format
                        'protein_pos': off_target_protein['protein_pos'].clone(),
                        'protein_element': off_target_protein['protein_element'].clone(),
                        'protein_is_backbone': off_target_protein['protein_is_backbone'].clone(),
                        'protein_atom_to_aa_type': off_target_protein['protein_atom_to_aa_type'].clone(),
                        'protein_id': torch.ones(len(off_target_protein['protein_element']), dtype=torch.long) * (i + 1),  # off-target ID = 1,2,3...
                        
                        # Use identical ligand data from on-target for perfect compatibility
                        'ligand_pos': data['ligand_pos'].clone(),
                        'ligand_element': data['ligand_element'].clone(),
                        'ligand_bond_index': data['ligand_bond_index'].clone(),
                        'ligand_bond_type': data['ligand_bond_type'].clone(),
                        'ligand_atom_feature': data['ligand_atom_feature'].clone(),
                        'ligand_hybridization': process_ligand_hybridization(data['ligand_hybridization']),
                        
                        # Affinity values
                        'affinity': torch.tensor(0.0, dtype=torch.float32),  # Dummy affinity
                        'id': i
                    }
                    
                    off_target_item = ProteinLigandData(**data_dict)
                    
                    # Add metadata
                    off_target_item.protein_filename = data.get('on_target_protein_filename', f'random_off_target_{i}')
                    off_target_item.ligand_filename = f"dummy_ligand_off_target_{i}"
                    
                    # Apply transform if provided
                    if transform is not None:
                        off_target_item = transform(off_target_item)
                    
                    off_target_data.append(off_target_item)
                    
                    print(f"Loaded random off-target {i+1}: {len(off_target_protein['protein_element'])} protein atoms")
                    
                except Exception as e:
                    print(f"Error loading off-target {i}: {e}")
                    continue
            
    finally:
        db.close()
    
    if off_target_data:
        print(f"Successfully loaded {len(off_target_data)} random off-target entries from LMDB")
        return off_target_data
    else:
        print("No off-target entries loaded")
        return None

def load_on_target_from_lmdb(lmdb_path, data_id, transform=None):
    """
    Load on-target protein pocket data directly from LMDB by data_id
    
    Args:
        lmdb_path (str): Path to LMDB database
        data_id (int): Index of the data entry to load
        transform: Data transformation to apply
        
    Returns:
        ProteinLigandData: On-target pocket data
    """
    import lmdb
    import pickle
    
    # Open LMDB database
    db = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with db.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            
            if data_id >= len(keys):
                raise IndexError(f"data_id {data_id} out of range (max: {len(keys)-1})")
                
            key = keys[data_id]
            data = pickle.loads(txn.get(key))
            
            # Extract on-target protein pocket data
            on_target_protein = data['on_target_protein']
            
            # Create ProteinLigandData for on-target (pocket only)
            data_dict = {
                # On-target protein pocket
                'protein_pos': on_target_protein['protein_pos'].clone(),
                'protein_element': on_target_protein['protein_element'].clone(),
                'protein_is_backbone': on_target_protein['protein_is_backbone'].clone(),
                'protein_atom_to_aa_type': on_target_protein['protein_atom_to_aa_type'].clone(),
                'protein_id': torch.zeros(len(on_target_protein['protein_element']), dtype=torch.long),  # on-target ID = 0
                
                # Original ligand data
                'ligand_pos': data['ligand_pos'].clone(),
                'ligand_element': data['ligand_element'].clone(),
                'ligand_bond_index': data['ligand_bond_index'].clone(),
                'ligand_bond_type': data['ligand_bond_type'].clone(),
                'ligand_atom_feature': data['ligand_atom_feature'].clone(),
                'ligand_hybridization': process_ligand_hybridization(data['ligand_hybridization']),
                
                # CRITICAL: Use actual affinity values from LMDB for proper guidance
                'on_target_affinity': torch.tensor(data['on_target_affinity'], dtype=torch.float32),
                'off_target_affinities': torch.tensor(data['off_target_affinities'], dtype=torch.float32),
                'affinity': torch.tensor(data['on_target_affinity'], dtype=torch.float32),
                'id': data_id
            }
            
            on_target_data = ProteinLigandData(**data_dict)
            
            # Add metadata
            on_target_data.protein_filename = data.get('on_target_protein_filename', '')
            on_target_data.ligand_filename = data.get('on_target_ligand_filename', '')
            
            # Apply transform if provided
            if transform is not None:
                on_target_data = transform(on_target_data)
            
            print(f"Loaded on-target pocket from LMDB (data_id={data_id}): {len(on_target_protein['protein_element'])} protein atoms, {len(data['ligand_element'])} ligand atoms")
            return on_target_data
            
    finally:
        db.close()
    
    return None

def load_validation_mapping():
    """Load validation to testset index mapping"""
    mapping_path = '/home/ktori1361/KGDiff/validation_testset_mapping.json'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return None

def load_validation_info():
    """Load validation protein information"""
    validation_path = '/home/ktori1361/KGDiff/multipro_validation_info.json'
    if os.path.exists(validation_path):
        with open(validation_path, 'r') as f:
            return json.load(f)
    return None

def load_safe_random_off_targets_from_lmdb(lmdb_path, on_target_data_id, num_off_targets=3, transform=None):
    """
    단백질 이름 기반으로 중복을 피하는 random off-target 로딩

    Args:
        lmdb_path (str): LMDB 경로
        on_target_data_id (int): On-target data ID (LMDB 인덱스)
        num_off_targets (int): 로드할 off-target 수
        transform: 데이터 변환 함수

    Returns:
        list: 안전한 off-target 데이터 리스트 (단백질 중복 없음)
    """
    validation_info = load_validation_info()
    if not validation_info:
        print("Warning: Could not load validation info, falling back to original method")
        return load_combined_off_targets_from_lmdb(lmdb_path, on_target_data_id, num_off_targets, transform)

    # On-target 단백질 이름 얻기
    on_target_validation_id = None
    on_target_protein_name = None
    for i, entry in enumerate(validation_info):
        if entry['idx'] == on_target_data_id:
            on_target_validation_id = i
            on_target_protein_name = entry['protein_dir']
            break

    if on_target_protein_name is None:
        print(f"Warning: Could not find on-target protein name for data_id {on_target_data_id}")
        return load_combined_off_targets_from_lmdb(lmdb_path, on_target_data_id, num_off_targets, transform)

    # 사용된 단백질 이름 추적 (on-target 포함)
    used_protein_names = {on_target_protein_name}
    available_validation_ids = []

    # 중복되지 않는 단백질들만 선택
    for i, entry in enumerate(validation_info):
        if i != on_target_validation_id:  # on-target 제외
            protein_name = entry['protein_dir']
            if protein_name not in used_protein_names:
                available_validation_ids.append(i)
                used_protein_names.add(protein_name)

    # Random 선택
    if len(available_validation_ids) < num_off_targets:
        print(f"Warning: Not enough unique proteins ({len(available_validation_ids)} available) for {num_off_targets} off-targets")
        num_off_targets = len(available_validation_ids)

    if num_off_targets == 0:
        print("Error: No unique off-target proteins available")
        return None

    selected_validation_ids = random.sample(available_validation_ids, num_off_targets)
    print(f"Selected unique off-target validation_ids: {selected_validation_ids}")

    # 선택된 validation_id들에 대한 data_id들 얻기
    selected_data_ids = []
    for val_id in selected_validation_ids:
        entry = validation_info[val_id]
        data_id = entry['idx']
        protein_name = entry['protein_dir']
        selected_data_ids.append(data_id)
        print(f"  Off-target validation_id {val_id}: {protein_name} (data_id: {data_id})")

    # 기존 load_combined_off_targets_from_lmdb 함수 활용하되, 선택된 data_id들 사용
    return load_combined_off_targets_with_specific_ids(lmdb_path, selected_data_ids, transform)

def load_combined_off_targets_with_specific_ids(lmdb_path, selected_data_ids, transform=None):
    """
    특정 data_id들로 off-target을 로드하는 함수

    Args:
        lmdb_path (str): LMDB 경로
        selected_data_ids (list): 선택된 data ID 리스트
        transform: 데이터 변환 함수

    Returns:
        list: 로드된 off-target 데이터 (combined 형태)
    """
    import lmdb
    import pickle

    if not selected_data_ids:
        return None

    try:
        env = lmdb.open(lmdb_path, readonly=True)

        with env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))

            off_target_proteins = []

            for data_id in selected_data_ids:
                if data_id < len(keys):
                    key = keys[data_id]
                    data = pickle.loads(txn.get(key))
                    off_target_protein = data['on_target_protein']
                    off_target_proteins.append(off_target_protein)
                    print(f"Loaded off-target {len(off_target_proteins)}: {len(off_target_protein['protein_element'])} protein atoms")
                else:
                    print(f"Warning: data_id {data_id} out of range")

            if not off_target_proteins:
                return None

            # Combine off-target proteins (기존 로직 재사용)
            combined_protein_pos = off_target_proteins[0]['protein_pos'].clone()
            combined_protein_element = off_target_proteins[0]['protein_element'].clone()
            combined_protein_is_backbone = off_target_proteins[0]['protein_is_backbone'].clone()
            combined_protein_atom_to_aa_type = off_target_proteins[0]['protein_atom_to_aa_type'].clone()
            combined_protein_id = torch.ones_like(off_target_proteins[0]['protein_element'], dtype=torch.long)

            # Combine additional off-targets
            for i, protein in enumerate(off_target_proteins[1:], 2):
                combined_protein_pos = torch.cat([combined_protein_pos, protein['protein_pos']], dim=0)
                combined_protein_element = torch.cat([combined_protein_element, protein['protein_element']], dim=0)
                combined_protein_is_backbone = torch.cat([combined_protein_is_backbone, protein['protein_is_backbone']], dim=0)
                combined_protein_atom_to_aa_type = torch.cat([combined_protein_atom_to_aa_type, protein['protein_atom_to_aa_type']], dim=0)
                protein_ids = torch.ones_like(protein['protein_element'], dtype=torch.long) * i
                combined_protein_id = torch.cat([combined_protein_id, protein_ids], dim=0)

            # Use ligand from first off-target (dummy ligand)
            first_data = pickle.loads(txn.get(keys[selected_data_ids[0]]))

            data_dict = {
                'protein_pos': combined_protein_pos,
                'protein_element': combined_protein_element,
                'protein_is_backbone': combined_protein_is_backbone,
                'protein_atom_to_aa_type': combined_protein_atom_to_aa_type,
                'protein_id': combined_protein_id,
                'ligand_pos': first_data['ligand_pos'].clone(),
                'ligand_element': first_data['ligand_element'].clone(),
                'ligand_bond_index': first_data['ligand_bond_index'].clone(),
                'ligand_bond_type': first_data['ligand_bond_type'].clone(),
                'ligand_atom_feature': first_data['ligand_atom_feature'].clone(),
                'ligand_hybridization': process_ligand_hybridization(first_data['ligand_hybridization']),
            }

            combined_off_target = ProteinLigandData(**data_dict)
            combined_off_target.protein_filename = f"safe_combined_off_targets_{len(off_target_proteins)}"
            combined_off_target.ligand_filename = "reference_ligand"

            if transform is not None:
                combined_off_target = transform(combined_off_target)

            return [combined_off_target]

    except Exception as e:
        print(f"Error loading safe random off-targets: {e}")
        return None

def validate_and_fix_selectivity_ids(on_target_id, off_target_ids, logger):
    """
    Selectivity 모드에서 중복 단백질 ID 검사 및 자동 교정
    dock_generated_ligands_unified.py와 동일한 로직 적용

    Args:
        on_target_id (int): On-target validation ID
        off_target_ids (list): Off-target validation IDs
        logger: Logger instance for output

    Returns:
        tuple: (corrected_on_target_id, corrected_off_target_ids)
    """
    try:
        from utils.protein_id_manager import validate_protein_ids, get_safe_protein_ids, get_protein_name_by_id

        # 중복 검사
        is_valid, errors = validate_protein_ids(on_target_id, off_target_ids)

        if not is_valid:
            logger.warning("=== PROTEIN DUPLICATION DETECTED IN SELECTIVITY MODE ===")
            for error in errors:
                logger.warning(f"  {error}")

            # 자동 교정
            safe_on_target, safe_off_targets = get_safe_protein_ids(on_target_id, off_target_ids)
            logger.info("=== AUTOMATIC CORRECTION ===")
            logger.info(f"Original: on_target={on_target_id}, off_targets={off_target_ids}")
            logger.info(f"Corrected: on_target={safe_on_target}, off_targets={safe_off_targets}")

            # 단백질 이름 표시
            logger.info(f"On-target protein: ID {safe_on_target} -> {get_protein_name_by_id(safe_on_target)}")
            for i, off_id in enumerate(safe_off_targets):
                logger.info(f"Off-target {i}: ID {off_id} -> {get_protein_name_by_id(off_id)}")

            return safe_on_target, safe_off_targets
        else:
            logger.info("✓ No protein duplications detected in selectivity mode")
            logger.info(f"On-target protein: ID {on_target_id} -> {get_protein_name_by_id(on_target_id)}")
            for i, off_id in enumerate(off_target_ids):
                logger.info(f"Off-target {i}: ID {off_id} -> {get_protein_name_by_id(off_id)}")
            return on_target_id, off_target_ids

    except ImportError as e:
        logger.warning(f"Could not import protein ID manager: {e}")
        logger.warning("Proceeding without duplication check - results may be inconsistent!")
        return on_target_id, off_target_ids

def find_validation_id_by_protein_name(lmdb_path, target_protein_name):
    """
    Find validation_id that matches the given protein name using pre-built mapping file
    
    Args:
        lmdb_path (str): Path to LMDB database (not used, kept for compatibility)
        target_protein_name (str): Protein name to search for (e.g., 'ACES_HUMAN_33_574_0')
        
    Returns:
        int: validation_id if found, None if not found
    """
    import json
    
    mapping_file = '/home/ktori1361/KGDiff/validation_testset_mapping.json'
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Get the list of validation proteins and find the index
        validation_proteins = mapping['validation_proteins']
        
        if target_protein_name in validation_proteins:
            validation_id = validation_proteins.index(target_protein_name)
            print(f"Found mapping: {target_protein_name} -> validation_id: {validation_id}")
            return validation_id
        else:
            print(f"Protein {target_protein_name} not found in validation set")
            # Try to find it by checking if protein appears multiple times
            indices = [i for i, protein in enumerate(validation_proteins) if protein == target_protein_name]
            if indices:
                validation_id = indices[0]  # Use first occurrence
                print(f"Found first occurrence: {target_protein_name} -> validation_id: {validation_id}")
                return validation_id
            
            print(f"Available validation proteins: {set(validation_proteins)}")
            return None
            
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        return None

def load_target_from_lmdb_by_validation_id(lmdb_path, validation_id, transform=None):
    """
    Load target protein data from LMDB using validation ID

    Args:
        lmdb_path (str): Path to LMDB database
        validation_id (int): Validation index (0-99)
        transform: Data transformation to apply

    Returns:
        ProteinLigandData: Target protein data or None if failed
    """
    try:
        # Load validation indices from split file
        split_file = './scratch2/data/multipro_final_ligand_aligned_split.pt'
        if not os.path.exists(split_file):
            print(f"Error: Split file not found: {split_file}")
            return None
            
        split_data = torch.load(split_file, map_location='cpu')
        val_indices = split_data['val']
        
        if validation_id >= len(val_indices):
            print(f"Error: validation_id {validation_id} out of range. Max: {len(val_indices)-1}")
            return None
            
        # Get actual LMDB index from validation split
        actual_lmdb_idx = val_indices[validation_id]
        print(f"Loading validation_id {validation_id}: LMDB index {actual_lmdb_idx}")
        
        # Load directly from LMDB using actual index
        import lmdb
        import pickle
        
        db = lmdb.open(lmdb_path, readonly=True, lock=False, create=False, max_readers=1)
        
        with db.begin() as txn:
            key = f'{actual_lmdb_idx:08d}'.encode()
            value = txn.get(key)
            
            if value is None:
                print(f"Error: Key {key} not found in LMDB")
                return None
                
            data = pickle.loads(value)
            
            # Extract on-target protein pocket data
            on_target_protein = data['on_target_protein']
            
            # Create ProteinLigandData for on-target (pocket only)
            data_dict = {
                # On-target protein pocket
                'protein_pos': on_target_protein['protein_pos'].clone(),
                'protein_element': on_target_protein['protein_element'].clone(),
                'protein_is_backbone': on_target_protein['protein_is_backbone'].clone(),
                'protein_atom_to_aa_type': on_target_protein['protein_atom_to_aa_type'].clone(),
                'protein_id': torch.zeros(len(on_target_protein['protein_element']), dtype=torch.long),
                
                # Original ligand data
                'ligand_pos': data['ligand_pos'].clone(),
                'ligand_element': data['ligand_element'].clone(),
                'ligand_bond_index': data['ligand_bond_index'].clone(),
                'ligand_bond_type': data['ligand_bond_type'].clone(),
                'ligand_atom_feature': data['ligand_atom_feature'].clone(),
                'ligand_hybridization': process_ligand_hybridization(data['ligand_hybridization']),
                
                # CRITICAL: Use actual affinity values from LMDB for proper guidance
                'on_target_affinity': torch.tensor(data['on_target_affinity'], dtype=torch.float32),
                'off_target_affinities': torch.tensor(data['off_target_affinities'], dtype=torch.float32),
            }
            
            # Apply transform
            protein_ligand_data = ProteinLigandData.from_dict(data_dict)
            if transform:
                protein_ligand_data = transform(protein_ligand_data)
            
            # Get protein name for logging
            protein_filename = data.get('on_target_protein_filename', 'unknown')
            if '/' in protein_filename:
                protein_name = protein_filename.split('/')[0]
            else:
                protein_name = protein_filename
                
            print(f"Successfully loaded validation_id {validation_id}: {protein_name}")
            
        db.close()
        return protein_ligand_data
        
    except Exception as e:
        print(f"Error loading validation_id {validation_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_individual_off_targets_from_lmdb(lmdb_path, on_target_val_id, off_target_val_ids=None, transform=None):
    """
    Load individual off-target protein data from LMDB for sequential selectivity mode
    
    Args:
        lmdb_path (str): Path to LMDB database  
        on_target_val_id (int): On-target validation ID (for reference only)
        off_target_val_ids (list): List of off-target validation IDs
        transform: Data transformation to apply
    
    Returns:
        list: List of individual ProteinLigandData objects for each off-target
    """
    print(f"Loading individual off-target data for sequential selectivity:")
    print(f"  On-target validation_id: {on_target_val_id} (skipped - using existing data)")
    if off_target_val_ids:
        print(f"  Off-target validation_ids: {off_target_val_ids}")
    else:
        print("  No off-target IDs specified")
        return []
    
    import lmdb
    import pickle
    
    # Load validation info to convert validation IDs to data IDs
    validation_info = load_validation_info()
    if not validation_info:
        print("Error: Validation info not found")
        return []
    
    # Convert validation IDs to data IDs
    off_target_data_list = []
    data_ids = []
    for val_id in off_target_val_ids:
        if val_id < len(validation_info):
            data_id = validation_info[val_id]['idx']
            protein_name = validation_info[val_id]['protein_dir']
            data_ids.append(data_id)
            print(f"  Off-target validation_id {val_id}: {protein_name} (data_id: {data_id})")
        else:
            print(f"Warning: validation_id {val_id} out of range")
    
    if not data_ids:
        return []
    
    # Open LMDB database
    db = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with db.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            
            # Load each off-target individually
            for data_id in data_ids:
                if data_id >= len(keys):
                    print(f"Warning: data_id {data_id} out of range")
                    continue
                
                key = keys[data_id]
                data = pickle.loads(txn.get(key))
                
                # Extract on-target protein pocket data (same as load_on_target_from_lmdb)
                on_target_protein = data['on_target_protein']
                
                # Create ProteinLigandData for off-target
                data_dict = {
                    # On-target protein pocket
                    'protein_pos': on_target_protein['protein_pos'].clone(),
                    'protein_element': on_target_protein['protein_element'].clone(),
                    'protein_is_backbone': on_target_protein['protein_is_backbone'].clone(),
                    'protein_atom_to_aa_type': on_target_protein['protein_atom_to_aa_type'].clone(),
                    'protein_id': torch.zeros(len(on_target_protein['protein_element']), dtype=torch.long),  # protein ID = 0
                    
                    # Original ligand data
                    'ligand_pos': data['ligand_pos'].clone(),
                    'ligand_element': data['ligand_element'].clone(),
                    'ligand_bond_index': data['ligand_bond_index'].clone(),
                    'ligand_bond_type': data['ligand_bond_type'].clone(),
                    'ligand_atom_feature': data['ligand_atom_feature'].clone(),
                    'ligand_hybridization': process_ligand_hybridization(data['ligand_hybridization']),
                    
                    # Affinity values
                    'on_target_affinity': torch.tensor(data['on_target_affinity'], dtype=torch.float32),
                    'off_target_affinities': torch.tensor(data['off_target_affinities'], dtype=torch.float32),
                    'affinity': torch.tensor(data['on_target_affinity'], dtype=torch.float32),
                    'id': data_id
                }
                
                off_target_data = ProteinLigandData(**data_dict)
                
                # Add metadata
                off_target_data.protein_filename = data.get('on_target_protein_filename', '')
                off_target_data.ligand_filename = data.get('on_target_ligand_filename', '')
                
                if transform is not None:
                    off_target_data = transform(off_target_data)
                
                off_target_data_list.append(off_target_data)
                print(f"    Loaded off-target data_id {data_id}")
    
    finally:
        db.close()
    
    print(f"Successfully loaded {len(off_target_data_list)} individual off-target proteins")
    return off_target_data_list

def load_multi_target_data_from_lmdb(lmdb_path, on_target_val_id, off_target_val_ids=None, transform=None):
    """
    MEMORY-OPTIMIZED: Load only off-target protein data from LMDB using validation IDs
    On-target data should already be available, so we don't load it again to save memory
    
    Args:
        lmdb_path (str): Path to LMDB database  
        on_target_val_id (int): On-target validation ID (for reference only)
        off_target_val_ids (list): List of off-target validation IDs (optional)
        transform: Data transformation to apply
    
    Returns:
        ProteinLigandData: Combined off-target data (or None if no off-targets)
    """
    print(f"Loading off-target data (memory-optimized):")
    print(f"  On-target validation_id: {on_target_val_id} (skipped - using existing data)")
    if off_target_val_ids:
        print(f"  Off-target validation_ids: {off_target_val_ids}")
    else:
        print("  No off-target IDs specified")
        return None
    
    # Load and combine off-targets directly like load_combined_off_targets_from_lmdb
    return load_specified_off_targets_combined(lmdb_path, off_target_val_ids, on_target_val_id, transform)

def load_off_targets_by_validation_ids(lmdb_path, off_target_val_ids, transform=None):
    """
    Load specified off-targets by validation IDs using direct LMDB access (no cursor iteration)

    Args:
        lmdb_path (str): Path to LMDB database
        off_target_val_ids (list): List of validation IDs to load
        transform: Data transformation to apply

    Returns:
        ProteinLigandData: Combined off-target data or None if failed
    """
    try:
        # Load validation indices from split file
        split_file = './scratch2/data/multipro_final_ligand_aligned_split.pt'
        if not os.path.exists(split_file):
            print(f"Error: Split file not found: {split_file}")
            return None
            
        split_data = torch.load(split_file, map_location='cpu')
        val_indices = split_data['val']
        
        # Open LMDB database
        import lmdb
        import pickle
        
        db = lmdb.open(lmdb_path, readonly=True, lock=False, create=False, max_readers=1)
        
        with db.begin() as txn:
            off_target_proteins = []
            reference_ligand = None
            
            for val_id in off_target_val_ids:
                if val_id >= len(val_indices):
                    print(f"Warning: validation_id {val_id} out of range")
                    continue
                
                # Get actual LMDB index
                actual_lmdb_idx = val_indices[val_id]
                key = f'{actual_lmdb_idx:08d}'.encode()
                value = txn.get(key)
                
                if value is None:
                    print(f"Warning: Key {key} not found in LMDB")
                    continue
                
                try:
                    data = pickle.loads(value)
                    off_target_protein = data['on_target_protein']
                    off_target_proteins.append(off_target_protein)
                    
                    # Use first off-target's ligand as reference
                    if reference_ligand is None:
                        reference_ligand = data
                    
                    # Get protein name for logging
                    protein_filename = data.get('on_target_protein_filename', 'unknown')
                    if '/' in protein_filename:
                        protein_name = protein_filename.split('/')[0]
                    else:
                        protein_name = protein_filename
                    
                    print(f"Loaded off-target validation_id {val_id}: {protein_name} ({len(off_target_protein['protein_element'])} atoms)")
                    
                except Exception as e:
                    print(f"Error loading validation_id {val_id}: {e}")
                    continue
        
        db.close()
        
        if not off_target_proteins or reference_ligand is None:
            print("No off-targets loaded successfully")
            return None
            
        # Combine off-target proteins into single graph
        combined_protein_pos = []
        combined_protein_element = []
        combined_protein_is_backbone = []
        combined_protein_atom_to_aa_type = []
        combined_protein_id = []
        
        current_protein_id = 1  # Start from 1 (0 is reserved for on-target)
        
        for off_protein in off_target_proteins:
            combined_protein_pos.append(off_protein['protein_pos'])
            combined_protein_element.append(off_protein['protein_element'])
            combined_protein_is_backbone.append(off_protein['protein_is_backbone'])
            combined_protein_atom_to_aa_type.append(off_protein['protein_atom_to_aa_type'])
            
            # Assign unique protein ID to each off-target
            protein_atoms = len(off_protein['protein_element'])
            combined_protein_id.append(torch.full((protein_atoms,), current_protein_id, dtype=torch.long))
            current_protein_id += 1
        
        # Create combined data structure
        data_dict = {
            # Combined off-target proteins
            'protein_pos': torch.cat(combined_protein_pos, dim=0),
            'protein_element': torch.cat(combined_protein_element, dim=0),
            'protein_is_backbone': torch.cat(combined_protein_is_backbone, dim=0),
            'protein_atom_to_aa_type': torch.cat(combined_protein_atom_to_aa_type, dim=0),
            'protein_id': torch.cat(combined_protein_id, dim=0),
            
            # Reference ligand data
            'ligand_pos': reference_ligand['ligand_pos'].clone(),
            'ligand_element': reference_ligand['ligand_element'].clone(),
            'ligand_bond_index': reference_ligand['ligand_bond_index'].clone(),
            'ligand_bond_type': reference_ligand['ligand_bond_type'].clone(),
            'ligand_atom_feature': reference_ligand['ligand_atom_feature'].clone(),
            'ligand_hybridization': process_ligand_hybridization(reference_ligand['ligand_hybridization']),
        }
        
        # Apply transform
        protein_ligand_data = ProteinLigandData.from_dict(data_dict)
        if transform:
            protein_ligand_data = transform(protein_ligand_data)
            
        print(f"Created combined off-target graph: {len(data_dict['protein_pos'])} total protein atoms")
        return protein_ligand_data
        
    except Exception as e:
        print(f"Error loading off-targets by validation IDs: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_specified_off_targets_combined(lmdb_path, off_target_val_ids, on_target_val_id, transform=None):
    """
    Load specified off-targets and combine them exactly like load_combined_off_targets_from_lmdb
    Uses the SAME structure to avoid memory/compatibility issues
    """
    import lmdb
    import pickle
    
    if not off_target_val_ids:
        return None
    
    # Load validation info to convert validation IDs to data IDs
    validation_info = load_validation_info()
    if not validation_info:
        print("Error: Validation info not found")
        return None
    
    # Convert on-target validation ID to data ID
    if on_target_val_id >= len(validation_info):
        print(f"Error: on_target validation_id {on_target_val_id} out of range")
        return None
    on_target_data_id = validation_info[on_target_val_id]['idx']
    
    # Convert validation IDs to data IDs
    data_ids = []
    for val_id in off_target_val_ids:
        if val_id < len(validation_info):
            data_id = validation_info[val_id]['idx']
            protein_name = validation_info[val_id]['protein_dir']
            data_ids.append(data_id)
            print(f"  Off-target validation_id {val_id}: {protein_name} (data_id: {data_id})")
        else:
            print(f"Warning: validation_id {val_id} out of range")
    
    if not data_ids:
        return None
    
    # Open LMDB database - use EXACT same pattern as load_combined_off_targets_from_lmdb
    db = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        with db.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            
            # Get reference ligand data from on-target (SAME as original)
            on_target_key = keys[on_target_data_id] if on_target_data_id < len(keys) else None
            on_target_data = pickle.loads(txn.get(on_target_key))
            
            # Collect off-target proteins (SAME structure as original)
            off_target_proteins = []
            
            for data_id in data_ids:
                if data_id >= len(keys):
                    print(f"Warning: data_id {data_id} out of range")
                    continue
                    
                try:
                    key = keys[data_id]
                    data = pickle.loads(txn.get(key))
                    off_target_protein = data['on_target_protein']
                    off_target_proteins.append(off_target_protein)
                    print(f"Loaded specified off-target: {len(off_target_protein['protein_element'])} protein atoms")
                except Exception as e:
                    print(f"Error loading off-target {data_id}: {e}")
                    continue
            
            if not off_target_proteins:
                return None
            
            # Combine all off-target proteins EXACTLY like the original function
            combined_protein_pos = off_target_proteins[0]['protein_pos'].clone()
            combined_protein_element = off_target_proteins[0]['protein_element'].clone()  
            combined_protein_is_backbone = off_target_proteins[0]['protein_is_backbone'].clone()
            combined_protein_atom_to_aa_type = off_target_proteins[0]['protein_atom_to_aa_type'].clone()
            combined_protein_id = torch.ones_like(off_target_proteins[0]['protein_element'], dtype=torch.long)  # First off-target = ID 1
            
            # Append other off-target proteins (SAME logic)
            for i, protein in enumerate(off_target_proteins[1:], 2):  # Start from ID 2
                combined_protein_pos = torch.cat([combined_protein_pos, protein['protein_pos']], dim=0)
                combined_protein_element = torch.cat([combined_protein_element, protein['protein_element']], dim=0)
                combined_protein_is_backbone = torch.cat([combined_protein_is_backbone, protein['protein_is_backbone']], dim=0)  
                combined_protein_atom_to_aa_type = torch.cat([combined_protein_atom_to_aa_type, protein['protein_atom_to_aa_type']], dim=0)
                protein_id_tensor = torch.full_like(protein['protein_element'], fill_value=i, dtype=torch.long)
                combined_protein_id = torch.cat([combined_protein_id, protein_id_tensor], dim=0)
            
            # Create combined off-target data object using reference ligand (SAME structure)
            # IMPORTANT: Include actual affinity information from LMDB
            data_dict = {
                # Combined protein attributes
                'protein_pos': combined_protein_pos,
                'protein_element': combined_protein_element,
                'protein_is_backbone': combined_protein_is_backbone,
                'protein_atom_to_aa_type': combined_protein_atom_to_aa_type,
                'protein_id': combined_protein_id,
                
                # Use identical ligand data from on-target for perfect compatibility
                'ligand_pos': on_target_data['ligand_pos'].clone(),
                'ligand_element': on_target_data['ligand_element'].clone(),
                'ligand_bond_index': on_target_data['ligand_bond_index'].clone(),
                'ligand_bond_type': on_target_data['ligand_bond_type'].clone(),
                'ligand_atom_feature': on_target_data['ligand_atom_feature'].clone(),
                'ligand_hybridization': process_ligand_hybridization(on_target_data['ligand_hybridization']),
                
                # CRITICAL: Include actual affinity values for guidance
                'on_target_affinity': torch.tensor(on_target_data['on_target_affinity'], dtype=torch.float32),
                'off_target_affinities': torch.tensor(on_target_data['off_target_affinities'], dtype=torch.float32),
                'affinity': torch.tensor(on_target_data['on_target_affinity'], dtype=torch.float32),  # For transforms
                
                # Add metadata for debugging
                'ligand_filename': f"specified_off_targets_{len(data_ids)}",
                'on_target_protein_filename': on_target_data.get('on_target_protein_filename', ''),
                'off_target_protein_filenames': on_target_data.get('off_target_protein_filenames', [])
            }
            
            combined_off_target_data = ProteinLigandData(**data_dict)
            
            # Apply transform if provided
            if transform is not None:
                combined_off_target_data = transform(combined_off_target_data)
            
            print(f"✅ Combined {len(off_target_proteins)} specified off-targets: {len(combined_protein_element)} total protein atoms")
            return combined_off_target_data
            
    finally:
        db.close()
    
    return None

def combine_off_targets_for_guidance(off_target_data_list):
    """
    Combine multiple off-target proteins into a single graph for guidance
    Similar to MultiProteinPairedDataset approach
    """
    if not off_target_data_list:
        return None
    
    if len(off_target_data_list) == 1:
        # Single off-target, protein_id should already be set
        return off_target_data_list[0]
    
    # Multiple off-targets: combine them
    combined_protein_pos = []
    combined_protein_element = []
    combined_protein_is_backbone = []
    combined_protein_atom_to_aa_type = []
    combined_protein_atom_feature = []
    combined_protein_id = []
    
    for i, off_data in enumerate(off_target_data_list):
        protein_id_value = i + 1  # Off-target IDs start from 1
        
        combined_protein_pos.append(off_data.protein_pos)
        combined_protein_element.append(off_data.protein_element)
        combined_protein_is_backbone.append(off_data.protein_is_backbone)
        combined_protein_atom_to_aa_type.append(off_data.protein_atom_to_aa_type)
        combined_protein_atom_feature.append(off_data.protein_atom_feature)
        
        # Assign unique protein ID to each off-target
        protein_ids = torch.full((len(off_data.protein_element),), protein_id_value, dtype=torch.long)
        combined_protein_id.append(protein_ids)
    
    # Concatenate all data
    combined_data = ProteinLigandData(
        protein_pos=torch.cat(combined_protein_pos, dim=0),
        protein_element=torch.cat(combined_protein_element, dim=0),
        protein_is_backbone=torch.cat(combined_protein_is_backbone, dim=0),
        protein_atom_to_aa_type=torch.cat(combined_protein_atom_to_aa_type, dim=0),
        protein_atom_feature=torch.cat(combined_protein_atom_feature, dim=0),
        protein_id=torch.cat(combined_protein_id, dim=0),
        
        # Use ligand data from first off-target (as template)
        ligand_element=off_target_data_list[0].ligand_element,
        ligand_pos=off_target_data_list[0].ligand_pos,
        ligand_bond_index=off_target_data_list[0].ligand_bond_index,
        ligand_bond_type=off_target_data_list[0].ligand_bond_type,
        ligand_atom_feature=off_target_data_list[0].ligand_atom_feature,
        ligand_filename=f"combined_off_targets_{len(off_target_data_list)}"
    )
    
    print(f"Combined {len(off_target_data_list)} off-target proteins into single graph: {len(combined_data.protein_element)} total atoms")
    return combined_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sampling.yml')
    parser.add_argument('-i', '--data_id', '--on_target_id', type=int, default=0, 
                       help='On-target validation ID (0-99). Legacy alias: data_id')
    parser.add_argument('--off_target_ids', type=int, nargs='*', default=None,
                       help='Off-target validation IDs (0-99). Example: --off_target_ids 1 2 3')
    parser.add_argument('--enable_selectivity', action='store_true', default=False,
                       help='Enable multi-target selectivity guidance (requires off_target_ids)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--guide_mode', type=str, default='joint', choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo', 'selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential', 'head1_head2_sequential'])  
    parser.add_argument('--type_grad_weight', type=float, default=0)
    parser.add_argument('--pos_grad_weight', type=float, default=0)
    # Head-specific guidance weights (overrides type_grad_weight and pos_grad_weight when specified)
    parser.add_argument('--head1_type_grad_weight', type=float, default=None,
                       help='Type gradient weight for head1 (interaction-based). If not specified, uses type_grad_weight')
    parser.add_argument('--head1_pos_grad_weight', type=float, default=None,
                       help='Position gradient weight for head1 (interaction-based). If not specified, uses pos_grad_weight')
    parser.add_argument('--head2_type_grad_weight', type=float, default=None,
                       help='Type gradient weight for head2 (no-interaction). If not specified, uses type_grad_weight')
    parser.add_argument('--head2_pos_grad_weight', type=float, default=None,
                       help='Position gradient weight for head2 (no-interaction). If not specified, uses pos_grad_weight')
    parser.add_argument('--on_target_only', action='store_true', default=False,
                       help='Only use on-target binding affinity guidance (ignore off-target guidance)')
    parser.add_argument('--use_lmdb_only', action='store_true', default=False,
                       help='Use pure LMDB approach like sample_diffusion_original2.py (overrides test_set_path)')
    parser.add_argument('--custom_lmdb', type=str, default=None,
                       help='Path to custom LMDB file (e.g., assignment_test_pockets.lmdb)')
    # off target guidance gradient scale
    parser.add_argument('--w_off', type=float, default=1.0, help='Weight for off-target penalty in selectivity score')
    parser.add_argument('--w_on', type=float, default=0.0, help='Weight for on-target affinity in selectivity score')
    parser.add_argument('--off_grad_weight', type=float, default=1.0, help='Weight for off-target gradient penalty in selectivity mode')
    parser.add_argument('--off_target_paths', type=str, nargs='*', default=None, help='Paths to off-target protein directories for selectivity mode')
    parser.add_argument('--mask_cross_protein', action='store_true', help='Force enable masking edges between different proteins')
    parser.add_argument('--no_mask_cross_protein', action='store_true', help='Force disable masking edges between different proteins (allow cross-protein interactions)')
    parser.add_argument('--result_path', type=str, default='./test_package')
    parser.add_argument('--test_set_path', type=str, default=None, help='Path to custom test set directory (e.g., data/multipro_validation_test_set)')
    parser.add_argument('--protein_name', type=str, default=None, help='Specific protein name to use from test set (overrides data_id)')
    parser.add_argument('--pocket_radius', type=float, default=10.0, help='Radius for extracting protein pocket atoms')
    
    # 인자 파싱
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()
    

     # 결과 저장 디렉토리 생성 및 설정 파일 복사
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    # 설정 파일 로드 및 시드 고정
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    # 가이던스 모드에 따라 적절한 사전 훈련된 모델(checkpoint) 로드

    # 'joint'(Default) denoising 네트워크와 expert 네트워크가 함께 훈련된 모델
    if args.guide_mode == 'joint':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    #pdbbind2020에서 샘플링함
    elif args.guide_mode == 'pdbbind_random':
        ckpt = torch.load(config.model['pdbbind_random'], map_location=args.device)
        value_ckpt = None
     # 학습된 신경망 대신 전통적인 분자 도킹 프로그램인 AutoDock Vina를 전문가(평가자)로 사용합니다.
    # 작동 방식: 생성 과정에서 만들어진 분자를 실시간으로 Vina를 이용해 단백질에 도킹시켜 결합 에너지(Vina score)를 계산합니다. 이 점수를 바탕으로 그래디언트를 추정하여 더 강하게 결합하는 분자 구조가 생성되도록 안내
    elif args.guide_mode == 'vina':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
     # 'valuenet'은 denoising 네트워크와 expert 네트워크가 별도로 훈련된 경우
    elif args.guide_mode == 'valuenet':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = torch.load(config.model['value_ckpt'], map_location=args.device)
     ## Guidence 안내없음, 그냥 Denoising Network만 사용해서 생성
    elif args.guide_mode == 'wo':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    # Selectivity mode: use joint checkpoint (similar to joint mode)
    elif args.guide_mode == 'selectivity':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    # Sequential selectivity mode: use joint checkpoint (similar to joint mode)
    elif args.guide_mode == 'sequential_selectivity':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    # Valuenet sequential selectivity mode: use separate policy and value checkpoints
    elif args.guide_mode == 'valuenet_sequential_selectivity':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = torch.load(config.model['value_ckpt'], map_location=args.device)
    # New pretrained joint aligned mode: use single joint model for memory efficiency
    elif args.guide_mode == 'pretrained_joint_aligned':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None  # Use same model for both policy and value
    # New pretrained joint no off-target interaction mode: use single joint model with no off-target interaction
    elif args.guide_mode == 'pretrained_joint_no_off_interaction':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None  # Use same model for both policy and value
    # New sequential on/off-target guidance mode
    elif args.guide_mode == 'joint_on_off_no_interaction_sequential':
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None  # Use same model for both policy and value
    # New head1-only sequential guidance mode
    elif args.guide_mode == 'head1_only_sequential':
        # This mode requires dual-head model with expert_pred_off (head2)
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None  # Use same model for both policy and value
    elif args.guide_mode == 'head2_only_sequential':
        # This mode requires dual-head model with expert_pred_off (head2)
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None  # Use same model for both policy and value
    else:
        raise NotImplementedError
    
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")

    # Fix data paths in checkpoint config to use new location
    from utils.path_fix import fix_checkpoint_data_paths
    fix_checkpoint_data_paths(ckpt, logger)

    # Transforms
     # 데이터 전처리 및 특징(feature) 추출을 위한 변환(transform) 정의
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    if args.custom_lmdb:
        # Custom LMDB file (e.g., assignment test pockets)
        logger.info(f"Using custom LMDB: {args.custom_lmdb}")
        from easydict import EasyDict
        custom_config = EasyDict({
            'name': 'pl',
            'path': args.custom_lmdb,
            'split': None  # No split file for custom LMDB
        })
        # Import lmdb and dataset classes
        import lmdb
        import pickle
        from datasets.pl_data import ProteinLigandData

        # Load custom LMDB directly
        db = lmdb.open(args.custom_lmdb, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with db.begin() as txn:
            num_examples = pickle.loads(txn.get(b'num_examples'))

        # Create simple dataset
        class CustomLMDBDataset:
            def __init__(self, lmdb_path, transform=None):
                self.db = lmdb.open(lmdb_path, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
                with self.db.begin() as txn:
                    self.num_examples = pickle.loads(txn.get(b'num_examples'))
                self.transform = transform

            def __len__(self):
                return self.num_examples

            def __getitem__(self, idx):
                with self.db.begin() as txn:
                    data = pickle.loads(txn.get(str(idx).encode()))
                if self.transform:
                    data = self.transform(data)
                return data

        test_set = CustomLMDBDataset(args.custom_lmdb, transform=transform)
        logger.info(f'Custom LMDB loaded: {len(test_set)} samples')

    elif args.use_lmdb_only:
        # Pure LMDB approach like sample_diffusion_original2.py
        logger.info("Using pure LMDB approach (like sample_diffusion_original2.py)")
        dataset, subsets = get_dataset(
            config=ckpt['config'].data,
            transform=transform
        )
        if ckpt['config'].data.name == 'pl':
            test_set = subsets['test']
        elif ckpt['config'].data.name == 'pdbbind':
            test_set = subsets['test']
        elif ckpt['config'].data.name == 'multipro':
            # Use validation set if test set is empty for multipro
            if len(subsets['test']) == 0:
                logger.info("Test set is empty, using validation set for multipro dataset")
                test_set = subsets['val']
            else:
                test_set = subsets['test']
        else:
            raise ValueError(f"Unsupported dataset for LMDB-only mode: {ckpt['config'].data.name}")
        logger.info(f'Pure LMDB test set loaded: {len(test_set)} samples')

    elif args.test_set_path is not None:
        # Use custom test set directory
        logger.info(f"Loading custom test set from: {args.test_set_path}")
        test_set = load_custom_test_set(args.test_set_path, transform=transform)
        logger.info(f'Custom test set loaded: {len(test_set)} samples')

        # Print available protein names for reference
        protein_names = [data.protein_name for data in test_set]
        logger.info(f"Available proteins: {protein_names[:10]}..." if len(protein_names) > 10 else f"Available proteins: {protein_names}")

    else:
        # Use original dataset loading for compatibility
        # For selectivity and sequential_selectivity modes, we'll load on-target data directly from LMDB
        if args.guide_mode in ['selectivity', 'sequential_selectivity']:
            logger.info("Loading on-target data directly from LMDB for selectivity mode...")
            # Create a dummy test_set with just the required entry
            test_set = [None]  # Placeholder, will be loaded later
        else:
            logger.info("Loading dataset from LMDB...")
            dataset, subsets = get_dataset(
                config=ckpt['config'].data,
                transform=transform
            )
            if ckpt['config'].data.name == 'pl':
                test_set = subsets['test']
            elif ckpt['config'].data.name == 'pdbbind':
                test_set = subsets['test']
            elif ckpt['config'].data.name == 'multipro':
                # Use validation set if test set is empty
                if len(subsets['test']) == 0:
                    test_set = subsets['val']
                    logger.info("Using validation set as test set is empty")
                else:
                    test_set = subsets['test']
            else:
                raise ValueError
            logger.info(f'Test: {len(test_set)}')


 # Load model
# 훈련된 가중치를 사용하여 모델(ScorePosNet3D) 초기화
    
    # Ensure center_ligand is available in model config (for backward compatibility)
    if hasattr(ckpt['config'], 'data') and hasattr(ckpt['config'].data, 'center_ligand'):
        if not hasattr(ckpt['config'].model, 'center_ligand'):
            ckpt['config'].model.center_ligand = ckpt['config'].data.center_ligand
    
    # Override center_ligand from sampling config if specified
    if hasattr(config.sample, 'center_ligand'):
        ckpt['config'].model.center_ligand = config.sample.center_ligand
        logger.info(f'Overriding model center_ligand to: {config.sample.center_ligand}')
    
    # Use checkpoint's featurizer dimensions (should match since we use checkpoint's config)
    logger.info(f"Protein feature dim: {protein_featurizer.feature_dim}")
    logger.info(f"Ligand feature dim: {ligand_featurizer.feature_dim}")
    
    # Fix architecture mismatch: determine if checkpoint uses original or current architecture
    import copy
    model_config = copy.deepcopy(ckpt['config'].model)

    # Check expert_pred architecture from checkpoint
    if 'model' in ckpt and 'expert_pred.0.weight' in ckpt['model']:
        expert_weight_shape = ckpt['model']['expert_pred.0.weight'].shape
        hidden_dim = getattr(model_config, 'hidden_dim', 128)

        logger.info(f"Config hidden_dim: {hidden_dim}")
        logger.info(f"Checkpoint expert_pred.0.weight shape: {expert_weight_shape}")

        # Determine architecture version based on expert_pred input dimension
        if expert_weight_shape[1] == hidden_dim:
            # Original architecture: expert_pred takes hidden_dim input
            logger.info("Detected ORIGINAL architecture (expert_pred input = hidden_dim)")
            model_config.use_original_expert = True
        elif expert_weight_shape[1] == hidden_dim * 2:
            # Current architecture: expert_pred takes hidden_dim*2 input
            logger.info("Detected CURRENT architecture (expert_pred input = hidden_dim*2)")
            model_config.use_original_expert = False
        else:
            logger.warning(f"Unknown expert_pred architecture. Input dim: {expert_weight_shape[1]}, hidden_dim: {hidden_dim}")
            model_config.use_original_expert = True  # Default to original

    # Check Head 2 architecture from checkpoint (for dual-head sam-pl models)
    if 'model' in ckpt:
        if 'cross_attn_query.weight' in ckpt['model']:
            # Atom-level cross-attention architecture
            logger.info("Detected ATOM-LEVEL CROSS-ATTENTION architecture (cross_attn_query/key/value/output layers found)")
            model_config.use_atom_level_cross_attn = True
            model_config.use_attention_head2 = False
        elif 'non_interaction_query.weight' in ckpt['model']:
            # Attention-based Head 2 (original training implementation)
            logger.info("Detected ATTENTION-BASED Head 2 (non_interaction_query/key/value layers found)")
            model_config.use_atom_level_cross_attn = False
            model_config.use_attention_head2 = True
        elif 'non_interaction_affinity_head.0.weight' in ckpt['model']:
            # Check the input dimension of non_interaction_affinity_head
            head2_weight_shape = ckpt['model']['non_interaction_affinity_head.0.weight'].shape
            hidden_dim = getattr(model_config, 'hidden_dim', 128)

            if head2_weight_shape[1] == hidden_dim:
                # Attention-based: input is hidden_dim (from attended features)
                logger.info("Detected ATTENTION-BASED Head 2 (input dim = hidden_dim)")
                model_config.use_atom_level_cross_attn = False
                model_config.use_attention_head2 = True
            elif head2_weight_shape[1] == hidden_dim * 2:
                # Concatenation-based: input is hidden_dim*2 (ligand + protein)
                logger.info("Detected CONCAT-BASED Head 2 (input dim = hidden_dim*2)")
                model_config.use_atom_level_cross_attn = False
                model_config.use_attention_head2 = False
            else:
                logger.warning(f"Unknown Head 2 architecture. Input dim: {head2_weight_shape[1]}, hidden_dim: {hidden_dim}")
                model_config.use_atom_level_cross_attn = False
                model_config.use_attention_head2 = False
        else:
            # No Head 2 detected, use default
            model_config.use_atom_level_cross_attn = False
            model_config.use_attention_head2 = False

    model = ScorePosNet3D(
        model_config,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    
    # Override cross-protein masking setting if provided via command line
    if args.mask_cross_protein:
        model.config.mask_cross_protein = True
        print(f"[INFO] Cross-protein masking ENABLED (--mask_cross_protein)")
    elif args.no_mask_cross_protein:
        model.config.mask_cross_protein = False
        print(f"[INFO] Cross-protein masking DISABLED (--no_mask_cross_protein)")
    else:
        # For selectivity mode, we want cross-protein edges masked to ensure independent evaluation
        if args.guide_mode == 'selectivity':
            model.config.mask_cross_protein = True
            print(f"[INFO] Cross-protein masking ENABLED for selectivity mode (ensuring independent protein evaluation)")
            print(f"[DEBUG] Model config mask_cross_protein set to: {model.config.mask_cross_protein}")
        else:
            # Safely check if mask_cross_protein exists in config
            mask_cross_protein_value = getattr(model.config, 'mask_cross_protein', False)
            print(f"[INFO] Using training config mask_cross_protein: {mask_cross_protein_value}")
        
        # Final verification of mask_cross_protein setting
        print(f"[FINAL] mask_cross_protein = {getattr(model.config, 'mask_cross_protein', 'NOT_SET')}")
    
     # 'valuenet' 모드인 경우, 별도의 전문가 네트워크(value_model) 로드
    if value_ckpt is not None:
        # Check value model's expert_pred architecture from checkpoint
        expert_pred_weight = value_ckpt['model']['expert_pred.0.weight']
        input_dim = expert_pred_weight.shape[1]
        logger.info(f"Value checkpoint expert_pred.0.weight shape: {expert_pred_weight.shape}")
        
        if input_dim == 128:
            logger.info("Detected ORIGINAL architecture for value model (expert_pred input = hidden_dim)")
            # Force original architecture for value model
            value_config = value_ckpt['config'].model
            value_config.use_original_expert = True
        else:
            logger.info("Detected CURRENT architecture for value model (expert_pred input = hidden_dim * 2)")
            value_config = value_ckpt['config'].model
            value_config.use_original_expert = False
        
        # value model
        value_model = ScorePosNet3D(
            value_config,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        value_model.load_state_dict(value_ckpt['model'])
    else:
        value_model = None
        
    # Select data based on protein name or data_id
    if args.use_lmdb_only:
        # Pure LMDB approach like sample_diffusion_original2.py
        logger.info(f"=== Pure LMDB Mode ===")
        logger.info(f"On-target data_id: {args.data_id}")

        # Validate on-target data_id
        if args.data_id >= len(test_set):
            logger.error(f"On-target data_id {args.data_id} is out of range. Test set has {len(test_set)} samples.")
            exit(1)

        # Load on-target data
        data = test_set[args.data_id]

        # Get on-target protein name
        on_target_protein = 'unknown'
        if hasattr(data, 'protein_filename'):
            on_target_protein = data.protein_filename.split('/')[0] if '/' in data.protein_filename else data.protein_filename[:10]
        elif hasattr(data, 'ligand_filename'):
            on_target_protein = data.ligand_filename.split('/')[0] if '/' in data.ligand_filename else data.ligand_filename[:10]

        logger.info(f"On-target protein: {on_target_protein} (data_id: {args.data_id})")

        # Log protein information if available
        if hasattr(data, 'ligand_filename'):
            logger.info(f"Ligand filename: {data.ligand_filename}")
        if hasattr(data, 'protein_filename'):
            logger.info(f"Protein filename: {data.protein_filename}")

        # Ensure on-target data has protein_id set to 0 for multi-target mode
        if not hasattr(data, 'protein_id') or data.protein_id is None:
            data.protein_id = torch.zeros(len(data.protein_element), dtype=torch.long)

        # Check if selectivity mode is requested with off-target IDs
        if args.enable_selectivity and args.off_target_ids and not args.on_target_only:
            logger.info(f"Off-target data_ids: {args.off_target_ids}")

            # Validate off-target data_ids
            valid_off_target_ids = []
            off_target_proteins = []

            for off_id in args.off_target_ids:
                if off_id >= len(test_set):
                    logger.warning(f"Off-target data_id {off_id} is out of range. Skipping.")
                    continue
                if off_id == args.data_id:
                    logger.warning(f"Off-target data_id {off_id} is same as on-target. Skipping.")
                    continue

                # Load off-target data to get protein name
                off_data = test_set[off_id]
                off_protein = 'unknown'
                if hasattr(off_data, 'protein_filename'):
                    off_protein = off_data.protein_filename.split('/')[0] if '/' in off_data.protein_filename else off_data.protein_filename[:10]
                elif hasattr(off_data, 'ligand_filename'):
                    off_protein = off_data.ligand_filename.split('/')[0] if '/' in off_data.ligand_filename else off_data.ligand_filename[:10]

                # Check if protein is different from on-target
                if off_protein == on_target_protein:
                    logger.warning(f"Off-target data_id {off_id} has same protein as on-target ({off_protein}). Skipping.")
                    continue

                valid_off_target_ids.append(off_id)
                off_target_proteins.append(off_protein)
                logger.info(f"Off-target protein: {off_protein} (data_id: {off_id})")

            if valid_off_target_ids:
                args.off_target_ids = valid_off_target_ids
                logger.info(f"Valid off-target data_ids: {valid_off_target_ids}")
                logger.info("Will use LMDB selectivity mode for multi-target generation")
            else:
                logger.warning("No valid off-target data_ids found. Switching to on-target only mode.")
                args.on_target_only = True
                args.enable_selectivity = False

    elif args.on_target_only:
        # ON-TARGET ONLY MODE: Load from LMDB using multipro validation set
        logger.info(f"Loading on-target data from LMDB for on-target only mode (validation_id={args.data_id})...")
        lmdb_path = './scratch2/data/multipro_final_protein_aligned.lmdb'
        
        # Use validation ID to load from multipro validation set
        data = load_target_from_lmdb_by_validation_id(lmdb_path, args.data_id, transform=transform)
        
        if data is None:
            logger.error(f"Failed to load on-target data from LMDB (validation_id={args.data_id})")
            logger.info("Please ensure --data_id corresponds to validation_id (0-99)")
            exit(1)
            
        # Log actual affinity values for verification
        logger.info(f"Loaded on-target affinity: {data.on_target_affinity.item():.4f}")
        logger.info("On-target only mode: Will use only this protein for binding affinity guidance")
        
    elif args.guide_mode in ['selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction']:
        # SELECTIVITY MODE: Use LMDB validation data for memory efficiency and real affinity values
        # Also support test_set_path for pretrained_joint_no_off_interaction

        # Check if test_set_path is provided (directory-based loading)
        if args.test_set_path is not None:
            logger.info(f"Using test_set directory for {args.guide_mode} mode")
            # Data already loaded from test_set in Line 1962, will use else branch below
            pass
        else:
            # Use LMDB validation set (default for these modes)
            logger.info(f"Loading on-target data from LMDB validation set (validation_id={args.data_id})...")

            # Use multipro LMDB with correct validation indices
            lmdb_path = './scratch2/data/multipro_final_ligand_aligned.lmdb'
            data = load_target_from_lmdb_by_validation_id(lmdb_path, args.data_id, transform=transform)

            if data is None:
                logger.error(f"Failed to load on-target data from LMDB (validation_id={args.data_id})")
                logger.info("Please ensure --data_id corresponds to validation_id (0-99)")
                exit(1)

            # Log actual affinity values for verification
            logger.info(f"Loaded on-target affinity: {data.on_target_affinity.item():.4f}")
            if hasattr(data, 'off_target_affinities') and len(data.off_target_affinities) > 0:
                off_affs = data.off_target_affinities.tolist()
                logger.info(f"Reference off-target affinities: {[f'{x:.4f}' for x in off_affs]}")
            else:
                logger.warning("No off-target affinity reference found in data")
                if not hasattr(data, 'off_target_affinities'):
                    data.off_target_affinities = torch.tensor([4.0], dtype=torch.float32)
            
    elif args.test_set_path is not None and args.protein_name is not None:
        # Find data by protein name
        data = None
        for test_data in test_set:
            if test_data.protein_name == args.protein_name:
                data = test_data
                logger.info(f"Found protein: {args.protein_name}")
                break

        if data is None:
            logger.error(f"Protein name '{args.protein_name}' not found in test set.")
            logger.info(f"Available proteins: {[d.protein_name for d in test_set[:10]]}{'...' if len(test_set) > 10 else ''}")
            exit(1)
    elif args.test_set_path is not None or (args.guide_mode in ['pretrained_joint_no_off_interaction', 'pretrained_joint_aligned'] and args.test_set_path is not None):
        # Use test_set directory (for pretrained_joint_no_off_interaction or when test_set_path is provided)
        if args.data_id >= len(test_set):
            logger.error(f"data_id {args.data_id} is out of range. Test set has {len(test_set)} samples.")
            logger.info(f"Using data_id 0 instead.")
            args.data_id = 0

        data = test_set[args.data_id]

        # Log which protein is being used
        if hasattr(data, 'protein_name'):
            logger.info(f"Using protein: {data.protein_name} (index {args.data_id})")
        else:
            logger.info(f"Using data_id: {args.data_id}")
    else:
        # Fallback: Use data_id from test_set if available
        if test_set and len(test_set) > 0:
            if args.data_id >= len(test_set):
                logger.error(f"data_id {args.data_id} is out of range. Test set has {len(test_set)} samples.")
                logger.info(f"Using data_id 0 instead.")
                args.data_id = 0

            data = test_set[args.data_id]

            # Log which protein is being used
            if hasattr(data, 'protein_name'):
                logger.info(f"Using protein: {data.protein_name} (index {args.data_id})")
            else:
                logger.info(f"Using data_id: {args.data_id}")
        else:
            logger.error("No data loaded. Please check your configuration.")
            exit(1)
    
    # Load off-target data for multi-target selectivity mode
    off_target_data = None
    off_target_data_list = []

    # Check if multi-target mode is enabled (excluding sequential modes which are handled separately)
    # Also support test_set_path for pretrained_joint_no_off_interaction
    # Also include use_lmdb_only mode
    # For use_lmdb_only, support both regular selectivity and sequential modes
    if args.use_lmdb_only:
        # For sequential modes, enable multi-target even without --enable_selectivity if off_target_ids are provided
        if args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
            use_multi_target = (args.off_target_ids is not None and len(args.off_target_ids) > 0)
        else:
            use_multi_target = (args.enable_selectivity and args.off_target_ids and
                               args.guide_mode in ['selectivity'])
    else:
        # Allow sequential modes to work with or without test_set_path
        if args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
            use_multi_target = (args.off_target_ids is not None and len(args.off_target_ids) > 0)
        else:
            use_multi_target = (args.enable_selectivity and args.off_target_ids and
                               args.guide_mode == 'selectivity' and
                               args.test_set_path is None)
    
    if use_multi_target:
        if args.use_lmdb_only:
            # Pure LMDB multi-target mode: data already loaded above
            logger.info(f"=== Pure LMDB Multi-Target Mode ===")
            logger.info(f"On-target data_id: {args.data_id}")
            logger.info(f"Off-target data_ids: {args.off_target_ids}")

            # Load off-target data from LMDB test set (same dataset as on-target)
            off_target_data_list = []
            for i, off_id in enumerate(args.off_target_ids):
                off_data = test_set[off_id]
                # Ensure protein_id is set for multi-target mode (start from 1, since on-target is 0)
                if not hasattr(off_data, 'protein_id') or off_data.protein_id is None:
                    off_data.protein_id = torch.ones(len(off_data.protein_element), dtype=torch.long) * (i + 1)
                off_target_data_list.append(off_data)

                # Get protein name for off-target
                try:
                    from utils.protein_id_manager import get_protein_name_by_id
                    off_protein_name = get_protein_name_by_id(off_id)
                    logger.info(f"Off-target protein: {off_protein_name} (data_id: {off_id})")
                except Exception as e:
                    logger.info(f"Loaded off-target data_id {off_id}")

                # Log off-target filenames if available
                if hasattr(off_data, 'ligand_filename'):
                    logger.info(f"  Ligand filename: {off_data.ligand_filename}")
                if hasattr(off_data, 'protein_filename'):
                    logger.info(f"  Protein filename: {off_data.protein_filename}")

                logger.info(f"Set protein_id to {i + 1} for off-target {off_id}")

            # Handle different guide modes for off-target data
            if args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                # Sequential modes: Use individual off-target data list
                off_target_data = off_target_data_list
                logger.info(f"Sequential mode ({args.guide_mode}): Using {len(off_target_data_list)} individual off-targets")
            else:
                # Regular selectivity: Combine off-targets for joint guidance
                if len(off_target_data_list) == 1:
                    off_target_data = off_target_data_list[0]
                else:
                    off_target_data = combine_off_targets_for_guidance(off_target_data_list)
                logger.info(f"Selectivity mode: Combined {len(off_target_data_list)} off-targets")

        else:
            # Original multipro validation approach
            logger.info(f"=== Multi-Target Mode Enabled (Memory-Optimized) ===")
            logger.info(f"On-target validation_id: {args.data_id}")
            logger.info(f"Off-target validation_ids: {args.off_target_ids}")

            # Use custom LMDB if specified, otherwise use default
            if args.custom_lmdb:
                lmdb_path = args.custom_lmdb
                logger.info(f"Using custom LMDB: {lmdb_path}")
                # For custom LMDB, load directly from test_set
                if args.data_id < len(test_set):
                    data = test_set[args.data_id]
                else:
                    logger.error(f"data_id {args.data_id} is out of range for custom LMDB")
                    exit(1)
            else:
                # Use same LMDB path as training config (same as selectivity mode)
                lmdb_path = './scratch2/data/multipro_final_protein_aligned.lmdb'
                # Load on-target data from LMDB (same as selectivity mode)
                logger.info(f"Loading on-target data from LMDB (validation_id={args.data_id})...")
                data = load_target_from_lmdb_by_validation_id(lmdb_path, args.data_id, transform=transform)

            if data is None:
                logger.error(f"Failed to load on-target data from LMDB (validation_id={args.data_id})")
                logger.info("Please ensure --data_id corresponds to validation_id (0-99)")
                exit(1)

        # Log actual affinity values for verification (skip for use_lmdb_only mode)
        if not args.use_lmdb_only:
            if hasattr(data, 'on_target_affinity') and data.on_target_affinity is not None:
                logger.info(f"Loaded on-target affinity: {data.on_target_affinity.item():.4f}")
            elif hasattr(data, 'affinity') and data.affinity is not None:
                logger.info(f"Loaded affinity: {data.affinity.item():.4f}")
            else:
                logger.info("No affinity data found in LMDB entry")

        # Only for non-use_lmdb_only mode (the original approach)
        if not args.use_lmdb_only:
            if hasattr(data, 'off_target_affinities') and len(data.off_target_affinities) > 0:
                off_affs = data.off_target_affinities.tolist()
                logger.info(f"Reference off-target affinities: {[f'{x:.4f}' for x in off_affs]}")
            else:
                logger.warning("No off-target affinity reference found in data")

            if args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                # Load individual off-target data for sequential processing
                if args.custom_lmdb:
                    # For custom LMDB, load directly from test_set
                    off_target_data = []
                    for off_id in args.off_target_ids:
                        if off_id < len(test_set):
                            off_target_data.append(test_set[off_id])
                        else:
                            logger.warning(f"Off-target ID {off_id} is out of range")
                else:
                    # For standard LMDB, use validation_id mapping
                    off_target_data = load_individual_off_targets_from_lmdb(
                        lmdb_path,
                        args.data_id,  # On-target validation ID (for reference only)
                        args.off_target_ids,  # Off-target validation IDs
                        transform=transform
                    )
            else:
                # Load combined off-target data for joint selectivity
                off_target_data = load_multi_target_data_from_lmdb(
                    lmdb_path,
                    args.data_id,  # On-target validation ID (for reference only)
                    args.off_target_ids,  # Off-target validation IDs
                    transform=transform
                )

        # Prepare off-target data for guidance
        if off_target_data:
            if args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
                logger.info(f"Loaded off-target proteins for sequential selectivity guidance (mode: {args.guide_mode})")
                # For sequential mode, we need to convert the combined off-target data to individual targets
                # off_target_data is already in the right format from load_multi_target_data_from_lmdb
                # It contains separate data objects for each off-target
            else:
                logger.info(f"Loaded combined off-target proteins for selectivity guidance")
                # Enable selectivity mode for non-sequential case
                args.guide_mode = 'selectivity'
        else:
            logger.warning("No off-target data loaded, disabling selectivity mode")
            if args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned']:
                logger.error(f"{args.guide_mode} mode requires off-target data")
            args.enable_selectivity = False
    
    elif args.guide_mode == 'selectivity' and not args.on_target_only and args.test_set_path is None:
        # Original selectivity mode (random off-targets) - ONLY if not on_target_only mode and no test_set_path
        logger.info(f"Loading off-target data randomly from LMDB (excluding on-target data_id: {args.data_id})")
        
        # Load randomly from LMDB and create combined graph like in training
        # Always use multipro_final_protein_aligned.lmdb for selectivity modes
        lmdb_path = './scratch2/data/multipro_final_protein_aligned.lmdb'
        off_target_data = load_safe_random_off_targets_from_lmdb(
            lmdb_path,
            args.data_id,  # Exclude on-target data_id
            num_off_targets=3,  # Use 3 off-targets as requested
            transform=transform
        )
        
        if off_target_data:
            logger.info(f"Loaded random off-target pockets from LMDB for selectivity")
        else:
            logger.warning("No off-target data loaded from LMDB, falling back to standard mode")
            args.guide_mode = 'joint'  # Fallback to joint mode
            
    elif args.guide_mode == 'selectivity' and not args.on_target_only and args.test_set_path is not None and args.off_target_ids:
        # Selectivity mode with test_set_path: Get protein names from test set, load data from LMDB
        logger.info(f"=== Selectivity Mode ===")
        logger.info(f"Original off-target validation_ids: {args.off_target_ids}")

        # PROTEIN ID VALIDATION AND SAFETY CHECK
        corrected_on_target_id, corrected_off_target_ids = validate_and_fix_selectivity_ids(
            args.data_id, args.off_target_ids, logger)
        args.data_id = corrected_on_target_id
        args.off_target_ids = corrected_off_target_ids

        logger.info(f"Final off-target validation_ids: {args.off_target_ids}")
        logger.info(f"Loading off-targets: Getting protein names from validation_ids {args.off_target_ids}, loading data from LMDB")
        
        off_target_validation_ids = []
        for off_id in args.off_target_ids:
            if off_id < len(test_set):
                test_off_data = test_set[off_id]
                off_protein_name = getattr(test_off_data, 'protein_name', f'protein_{off_id}')
                logger.info(f"Off-target {off_id}: {off_protein_name}")
                
                # Find corresponding validation_id in LMDB
                lmdb_path = './scratch2/data/multipro_final_protein_aligned.lmdb'
                matching_val_id = find_validation_id_by_protein_name(lmdb_path, off_protein_name)
                
                if matching_val_id is not None:
                    off_target_validation_ids.append(matching_val_id)
                    logger.info(f"Found off-target validation_id: {matching_val_id} for {off_protein_name}")
                else:
                    logger.warning(f"No matching LMDB entry for off-target: {off_protein_name}")
            else:
                logger.warning(f"Off-target ID {off_id} is out of range (test set has {len(test_set)} samples)")
        
        if off_target_validation_ids:
            # Load off-targets from LMDB using validation_ids (memory efficient)
            lmdb_path = './scratch2/data/multipro_final_protein_aligned.lmdb'
            # Load off-targets using direct access to avoid LMDB corruption
            off_target_data = load_off_targets_by_validation_ids(
                lmdb_path, 
                off_target_validation_ids,  # Use specified validation IDs
                transform=transform
            )
            
            if off_target_data:
                logger.info(f"Loaded {len(off_target_validation_ids)} off-targets from LMDB for selectivity guidance")
            else:
                logger.warning("Failed to load off-targets from LMDB, falling back to standard mode")
                off_target_data = None
                args.guide_mode = 'joint'
        else:
            logger.warning("No valid off-targets found, falling back to standard mode")
            off_target_data = None
            args.guide_mode = 'joint'
    elif args.guide_mode == 'selectivity' and args.on_target_only:
        # ON-TARGET ONLY mode with selectivity guide_mode: Don't load off-targets
        logger.info("On-target only mode: Skipping off-target loading (will use only on-target guidance)")
        off_target_data = None
    
    elif args.guide_mode in ['sequential_selectivity', 'valuenet_sequential_selectivity', 'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential']:
        # Sequential selectivity modes: Load specific off-targets for sequential processing
        if args.off_target_ids:
            logger.info(f"=== {args.guide_mode.title()} Mode ===")
            logger.info(f"Original off-target validation_ids: {args.off_target_ids}")

            # PROTEIN ID VALIDATION AND SAFETY CHECK
            corrected_on_target_id, corrected_off_target_ids = validate_and_fix_selectivity_ids(
                args.data_id, args.off_target_ids, logger)
            args.data_id = corrected_on_target_id
            args.off_target_ids = corrected_off_target_ids

            logger.info(f"Final off-target validation_ids: {args.off_target_ids}")

            # Check if using test_set_path (directory-based) or LMDB
            if args.test_set_path is not None:
                # Use test_set directory: Load off-targets from test_set
                logger.info(f"Loading off-targets from test_set directory")
                off_target_data = []
                for off_target_id in args.off_target_ids:
                    if off_target_id < len(test_set):
                        off_target = test_set[off_target_id]
                        off_target_data.append(off_target)
                        off_protein_name = getattr(off_target, 'protein_name', f'protein_{off_target_id}')
                        logger.info(f"Loaded off-target from test_set: {off_protein_name} (index {off_target_id})")
                    else:
                        logger.warning(f"Off-target ID {off_target_id} is out of range (test set has {len(test_set)} samples)")

                if not off_target_data:
                    logger.warning("No valid off-targets found in test_set")
                    off_target_data = None
            else:
                # Use LMDB validation set (default or custom)
                # Check if custom LMDB is specified, otherwise use default
                if args.custom_lmdb:
                    lmdb_path = args.custom_lmdb
                    logger.info(f"Using custom LMDB for off-targets: {lmdb_path}")
                else:
                    lmdb_path = './scratch2/data/multipro_final_ligand_aligned.lmdb'

                # Load individual off-targets using same method as on-target
                off_target_data = []
                for off_target_id in args.off_target_ids:
                    # For custom LMDB, use direct indexing instead of validation_id mapping
                    if args.custom_lmdb:
                        # Custom LMDB: direct index access
                        if off_target_id < len(test_set):
                            off_target = test_set[off_target_id]
                        else:
                            logger.warning(f"Off-target ID {off_target_id} is out of range")
                            off_target = None
                    else:
                        # Standard LMDB: use validation_id mapping
                        off_target = load_target_from_lmdb_by_validation_id(lmdb_path, off_target_id, transform=transform)
                    if off_target is not None:
                        # Extract only protein data (remove ligand for off-target)
                        off_target_protein_data = {
                            'protein_atom_feature': off_target.protein_atom_feature,
                            'protein_pos': off_target.protein_pos,
                            'protein_element_batch': getattr(off_target, 'protein_element_batch',
                                                           torch.zeros_like(off_target.protein_pos[:, 0], dtype=torch.long))
                        }
                        off_target_data.append(off_target_protein_data)

            if off_target_data:
                logger.info(f"Loaded {len(off_target_data)} off-target proteins for {args.guide_mode} guidance")
            else:
                logger.error(f"{args.guide_mode} mode requires off-target data")
                exit(1)
        else:
            # Allow pretrained_joint_aligned, pretrained_joint_no_off_interaction, and joint_on_off_no_interaction_sequential modes without off-target IDs (on-target only guidance)
            if args.guide_mode in ['pretrained_joint_aligned', 'pretrained_joint_no_off_interaction', 'joint_on_off_no_interaction_sequential']:
                logger.info(f"=== {args.guide_mode.title()} Mode (On-Target Only) ===")
                logger.info("No off-target IDs provided, using on-target only guidance")
                off_target_data = None  # Set to None to indicate on-target only mode
            else:
                logger.error(f"{args.guide_mode} mode requires --off_target_ids")
                exit(1)
    
    # Set device and batch size
    effective_device = args.device
    effective_batch_size = args.batch_size

    # Set debug log file for model
    debug_log_path = os.path.join(result_path, 'debug_sampling.log')
    model._debug_log_file = debug_log_path

    # Initialize debug log file with header
    with open(debug_log_path, 'w') as f:
        f.write(f"=== Debug Sampling Log ===\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Guide mode: {args.guide_mode}\n")
        f.write(f"Data ID: {args.data_id}\n")
        f.write(f"Off-target IDs: {args.off_target_ids}\n")
        f.write(f"w_on: {args.w_on}, w_off: {args.w_off}\n")
        f.write(f"type_grad_weight: {args.type_grad_weight}, pos_grad_weight: {args.pos_grad_weight}\n")
        # Log head-specific weights if specified
        if args.head1_type_grad_weight is not None or args.head1_pos_grad_weight is not None:
            f.write(f"head1_type_grad_weight: {args.head1_type_grad_weight}, head1_pos_grad_weight: {args.head1_pos_grad_weight}\n")
        if args.head2_type_grad_weight is not None or args.head2_pos_grad_weight is not None:
            f.write(f"head2_type_grad_weight: {args.head2_type_grad_weight}, head2_pos_grad_weight: {args.head2_pos_grad_weight}\n")
        f.write(f"=" * 80 + "\n\n")

    logger.info(f"Debug output will be saved to: {debug_log_path}")

    result_tuple = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=effective_batch_size, device=effective_device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        center_ligand=getattr(config.sample, 'center_ligand', False),
        sample_num_atoms=config.sample.sample_num_atoms,
        guide_mode=args.guide_mode,
        value_model=value_model,
        type_grad_weight=args.type_grad_weight,
        pos_grad_weight=args.pos_grad_weight,
        w_off=args.w_off,
        w_on=args.w_on,
        off_target_data=off_target_data,
        off_grad_weight=args.off_grad_weight,
        on_target_only=args.on_target_only,
        head1_type_grad_weight=args.head1_type_grad_weight,
        head1_pos_grad_weight=args.head1_pos_grad_weight,
        head2_type_grad_weight=args.head2_type_grad_weight,
        head2_pos_grad_weight=args.head2_pos_grad_weight
    )
    
    # Unpack results (handle both old and new return formats)
    if len(result_tuple) == 14:  # Newest format with on-target and off-target data
        (pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj,
         pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, pred_exp_off, pred_exp_off_traj,
         pred_exp_on, pred_exp_on_traj, time_list) = result_tuple
        
    elif len(result_tuple) == 12:  # Old format with off-target data only
        (pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj,
         pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, pred_exp_off, pred_exp_off_traj, time_list) = result_tuple
        pred_exp_on, pred_exp_on_traj = [], []

    else:  # Oldest format (backward compatibility)
        (pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj,
         pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list) = result_tuple
        pred_exp_off, pred_exp_off_traj = [], []
        pred_exp_on, pred_exp_on_traj = [], []
    
    ## 기본 딕셔너리 생성 (모든 Guidance 모드에서 공통적으로 필요한 필드)
    # 생성된 결과들을 딕셔너리 형태로 정리
    result = {
        'data': data, # 사용된 단백질 데이터
        'pred_ligand_pos': pred_pos, #최종 생성된 리간드 원자좌표
        'pred_ligand_v': pred_v, #최종 생성된 리간드 원자유형
        'pred_exp': pred_exp, # 최종 예측된 결합 친화도 (일반 joint 모드는 단일 친화도, Selectivity 모드에서는 Selectivity Score(v_on - v_off))
        'pred_ligand_pos_traj': pred_pos_traj, #위치 생성 궤적
        'pred_ligand_v_traj': pred_v_traj, # 유형 생성 궤적
        'pred_exp_traj': pred_exp_traj, #결합 친화도 예측 궤적
        'pred_exp_atom_traj': pred_exp_atom_traj,
        'time': time_list
    }
    
    # Add off-target and on-target results for selectivity modes
    if args.guide_mode in ['selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity',
                           'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction',
                           'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
        if len(pred_exp_off) > 0 or len(pred_exp_off_traj) > 0:
            result['pred_exp_off'] = pred_exp_off  # Final off-target predictions
            result['pred_exp_off_traj'] = pred_exp_off_traj  # Off-target prediction trajectory
        if len(pred_exp_on) > 0 or len(pred_exp_on_traj) > 0:
            result['pred_exp_on'] = pred_exp_on  # Final on-target predictions
            result['pred_exp_on_traj'] = pred_exp_on_traj  # On-target prediction trajectory
        if off_target_data is not None:
            result['off_target_data'] = off_target_data  # Reference to off-target data used
    logger.info('Sample done!')

    # 최종결과를 파일로 저장
    # Use on-target ID for simple, consistent filenames
    result_filename = f'result_{args.data_id}.pt'

    torch.save(result, os.path.join(result_path, result_filename))
    logger.info(f'Results saved to: {result_filename} (on-target ID: {args.data_id})')

    # Automatically generate trajectory visualizations for selectivity modes
    if args.guide_mode in ['selectivity', 'sequential_selectivity', 'valuenet_sequential_selectivity',
                           'pretrained_joint_aligned', 'pretrained_joint_no_off_interaction',
                           'joint_on_off_no_interaction_sequential', 'head1_only_sequential', 'head2_only_sequential']:
        if 'pred_exp_on_traj' in result and 'pred_exp_off_traj' in result:
            try:
                logger.info('Generating affinity trajectory visualizations...')
                from pathlib import Path
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                import numpy as np

                # Extract trajectories
                exp_on_traj = result['pred_exp_on_traj']
                exp_off_traj = result['pred_exp_off_traj']

                # Convert to numpy if needed
                if torch.is_tensor(exp_on_traj):
                    exp_on_traj = exp_on_traj.cpu().numpy()
                if torch.is_tensor(exp_off_traj):
                    exp_off_traj = exp_off_traj.cpu().numpy()

                # Reshape to [num_steps, num_samples]
                if exp_on_traj.ndim == 3:
                    exp_on_traj = exp_on_traj.squeeze(-1)
                if exp_off_traj.ndim == 3:
                    exp_off_traj = exp_off_traj.squeeze(-1)

                # Get actual number of generated samples from trajectory shape
                num_steps = exp_on_traj.shape[0]
                actual_num_samples = exp_on_traj.shape[1] if exp_on_traj.ndim > 1 else 1
                timesteps = np.arange(num_steps)

                # Verify off-target trajectory has same number of samples as on-target
                num_off_samples = exp_off_traj.shape[1] if exp_off_traj.ndim > 1 else 1
                if num_off_samples != actual_num_samples:
                    logger.warning(f'Off-target samples ({num_off_samples}) != on-target samples ({actual_num_samples})')
                    # Use minimum to avoid index errors
                    actual_num_samples = min(actual_num_samples, num_off_samples)

                # Limit visualization to first 10 samples
                max_vis_samples = min(actual_num_samples, 10)
                if actual_num_samples > 10:
                    logger.info(f'Limiting visualization to first 10 samples (out of {actual_num_samples} generated)')

                # Create visualization directory
                vis_dir = Path(result_path) / 'visualizations'
                vis_dir.mkdir(exist_ok=True)

                # Plot 1: Separate trajectories with dual-scale Y-axis labels
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))

                # On-Target plot
                ax = axes[0]
                for i in range(max_vis_samples):
                    ax.plot(timesteps, exp_on_traj[:, i], label=f'Sample {i+1}', alpha=0.7, linewidth=2)
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Normalized Affinity (Vina kcal/mol)', fontsize=12)
                ax.set_title('On-Target Binding Affinity Evolution During Sampling', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                # Set Y-axis range based on actual data
                y_min_on = exp_on_traj[:, :max_vis_samples].min()
                y_max_on = exp_on_traj[:, :max_vis_samples].max()
                y_range_on = y_max_on - y_min_on
                y_min_padded = max(0, y_min_on - 0.05 * y_range_on)  # Add 5% padding
                y_max_padded = min(1, y_max_on + 0.05 * y_range_on)
                ax.set_ylim(y_min_padded, y_max_padded)
                # Create ticks in the actual data range
                yticks_on = np.linspace(y_min_padded, y_max_padded, 6)
                ax.set_yticks(yticks_on)
                yticklabels_on = [f'{norm:.2f} ({-norm*16:.1f})' for norm in yticks_on]
                ax.set_yticklabels(yticklabels_on)

                # Off-Target plot
                ax = axes[1]
                for i in range(max_vis_samples):
                    ax.plot(timesteps, exp_off_traj[:, i], label=f'Sample {i+1}', alpha=0.7, linewidth=2)
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Normalized Affinity (Vina kcal/mol)', fontsize=12)
                ax.set_title('Off-Target Binding Affinity Evolution During Sampling', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                # Set Y-axis range based on actual data
                y_min_off = exp_off_traj[:, :max_vis_samples].min()
                y_max_off = exp_off_traj[:, :max_vis_samples].max()
                y_range_off = y_max_off - y_min_off
                y_min_padded = max(0, y_min_off - 0.05 * y_range_off)  # Add 5% padding
                y_max_padded = min(1, y_max_off + 0.05 * y_range_off)
                ax.set_ylim(y_min_padded, y_max_padded)
                # Create ticks in the actual data range
                yticks_off = np.linspace(y_min_padded, y_max_padded, 6)
                ax.set_yticks(yticks_off)
                yticklabels_off = [f'{norm:.2f} ({-norm*16:.1f})' for norm in yticks_off]
                ax.set_yticklabels(yticklabels_off)

                plt.tight_layout()
                plt.savefig(vis_dir / 'affinity_trajectories_separate.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Plot 2: Selectivity trajectory
                fig, ax = plt.subplots(figsize=(12, 6))
                all_selectivity_values = []
                for i in range(max_vis_samples):
                    selectivity = exp_on_traj[:, i] - exp_off_traj[:, i]
                    ax.plot(timesteps, selectivity, label=f'Sample {i+1}', linewidth=2.5, alpha=0.8)
                    all_selectivity_values.extend([selectivity.min(), selectivity.max()])

                ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero selectivity')
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Selectivity (On - Off) (Vina kcal/mol)', fontsize=12)
                ax.set_title('Selectivity Evolution During Sampling\n(Positive = On-target preferred, Negative = Off-target preferred)',
                             fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Auto-scale Y-axis
                y_min_sel = min(all_selectivity_values)
                y_max_sel = max(all_selectivity_values)
                y_range_sel = y_max_sel - y_min_sel
                y_min_padded_sel = y_min_sel - 0.05 * y_range_sel
                y_max_padded_sel = y_max_sel + 0.05 * y_range_sel
                ax.set_ylim(y_min_padded_sel, y_max_padded_sel)
                yticks_sel = np.linspace(y_min_padded_sel, y_max_padded_sel, 6)
                ax.set_yticks(yticks_sel)
                yticklabels_sel = [f'{sel:.3f} ({-sel*16:.2f})' for sel in yticks_sel]
                ax.set_yticklabels(yticklabels_sel)

                # Add colored background regions after setting ylim
                ax.axhspan(0, y_max_padded_sel, alpha=0.1, color='green')
                ax.axhspan(y_min_padded_sel, 0, alpha=0.1, color='red')

                plt.tight_layout()
                plt.savefig(vis_dir / 'selectivity_trajectory.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Plot 3: Comparison plot
                # Use grid layout for more samples (2 columns)
                if max_vis_samples <= 5:
                    # Vertical layout for few samples
                    fig, axes = plt.subplots(max_vis_samples, 1, figsize=(12, 4*max_vis_samples))
                    if max_vis_samples == 1:
                        axes = [axes]
                else:
                    # Grid layout for many samples
                    n_cols = 2
                    n_rows = (max_vis_samples + n_cols - 1) // n_cols  # Ceiling division
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
                    axes = axes.flatten()

                for i in range(max_vis_samples):
                    ax = axes[i]
                    line_on = ax.plot(timesteps, exp_on_traj[:, i], label='On-target', color='blue', linewidth=2.5, alpha=0.8)
                    line_off = ax.plot(timesteps, exp_off_traj[:, i], label='Off-target', color='red', linewidth=2.5, alpha=0.8)

                    selectivity = exp_on_traj[:, i] - exp_off_traj[:, i]
                    ax2 = ax.twinx()
                    line_sel = ax2.plot(timesteps, selectivity, label='Selectivity (On - Off)',
                                       color='green', linewidth=2, alpha=0.6, linestyle='--')
                    ax2.set_ylabel('Selectivity', fontsize=11, color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.3)

                    ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                    ax.set_ylabel('Normalized Affinity (Vina kcal/mol)', fontsize=11)
                    ax.set_title(f'Sample {i+1}: On-Target vs Off-Target Affinity Evolution', fontsize=13, fontweight='bold')

                    # Set Y-axis range based on actual data for this sample
                    y_min_sample = min(exp_on_traj[:, i].min(), exp_off_traj[:, i].min())
                    y_max_sample = max(exp_on_traj[:, i].max(), exp_off_traj[:, i].max())
                    y_range_sample = y_max_sample - y_min_sample
                    y_min_padded = max(0, y_min_sample - 0.05 * y_range_sample)
                    y_max_padded = min(1, y_max_sample + 0.05 * y_range_sample)
                    ax.set_ylim(y_min_padded, y_max_padded)
                    # Create ticks with dual labels
                    yticks_sample = np.linspace(y_min_padded, y_max_padded, 6)
                    ax.set_yticks(yticks_sample)
                    yticklabels_sample = [f'{norm:.2f} ({-norm*16:.1f})' for norm in yticks_sample]
                    ax.set_yticklabels(yticklabels_sample)

                    lines = line_on + line_off + line_sel
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper left')
                    ax.grid(True, alpha=0.3)

                    textstr = f'Initial: On={exp_on_traj[0, i]:.4f}, Off={exp_off_traj[0, i]:.4f}, Sel={selectivity[0]:.4f}\n'
                    textstr += f'Final:   On={exp_on_traj[-1, i]:.4f}, Off={exp_off_traj[-1, i]:.4f}, Sel={selectivity[-1]:.4f}\n'
                    textstr += f'Change:  ΔOn={exp_on_traj[-1, i]-exp_on_traj[0, i]:.4f}, ΔOff={exp_off_traj[-1, i]-exp_off_traj[0, i]:.4f}'
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Hide unused subplots if grid layout is used
                if max_vis_samples > 5:
                    for i in range(max_vis_samples, len(axes)):
                        axes[i].set_visible(False)

                plt.tight_layout()
                plt.savefig(vis_dir / 'affinity_trajectories_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Plot 4: Change analysis
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                all_cumulative_values = []
                for i in range(max_vis_samples):
                    cumulative_on = exp_on_traj[:, i] - exp_on_traj[0, i]
                    cumulative_off = exp_off_traj[:, i] - exp_off_traj[0, i]
                    ax.plot(timesteps, cumulative_on, label=f'On-target Sample {i+1}', linewidth=2, linestyle='-')
                    ax.plot(timesteps, cumulative_off, label=f'Off-target Sample {i+1}', linewidth=2, linestyle='--')
                    all_cumulative_values.extend([cumulative_on.min(), cumulative_on.max(),
                                                 cumulative_off.min(), cumulative_off.max()])
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Cumulative Affinity Change (Vina kcal/mol)', fontsize=12)
                ax.set_title('Cumulative Affinity Change from Initial State', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                # Auto-scale Y-axis
                y_min_cum = min(all_cumulative_values)
                y_max_cum = max(all_cumulative_values)
                y_range_cum = y_max_cum - y_min_cum
                y_min_padded_cum = y_min_cum - 0.05 * y_range_cum
                y_max_padded_cum = y_max_cum + 0.05 * y_range_cum
                ax.set_ylim(y_min_padded_cum, y_max_padded_cum)
                yticks_cum = np.linspace(y_min_padded_cum, y_max_padded_cum, 6)
                ax.set_yticks(yticks_cum)
                yticklabels_cum = [f'{delta:.2f} ({-delta*16:.1f})' for delta in yticks_cum]
                ax.set_yticklabels(yticklabels_cum)

                ax = axes[1]
                window = 100
                all_rate_values = []
                for i in range(max_vis_samples):
                    rate_on = np.gradient(exp_on_traj[:, i])
                    rate_off = np.gradient(exp_off_traj[:, i])
                    rate_on_smooth = np.convolve(rate_on, np.ones(window)/window, mode='same')
                    rate_off_smooth = np.convolve(rate_off, np.ones(window)/window, mode='same')
                    ax.plot(timesteps, rate_on_smooth, label=f'On-target Sample {i+1}', linewidth=2, linestyle='-')
                    ax.plot(timesteps, rate_off_smooth, label=f'Off-target Sample {i+1}', linewidth=2, linestyle='--')
                    all_rate_values.extend([rate_on_smooth.min(), rate_on_smooth.max(),
                                           rate_off_smooth.min(), rate_off_smooth.max()])
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
                ax.set_xlabel('Iteration (Timestep 999→0)', fontsize=12)
                ax.set_ylabel('Rate of Affinity Change (Vina kcal/mol per step)', fontsize=12)
                ax.set_title(f'Rate of Affinity Change (Moving Avg, window={window})', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                # Auto-scale Y-axis
                y_min_rate = min(all_rate_values)
                y_max_rate = max(all_rate_values)
                y_range_rate = y_max_rate - y_min_rate
                y_min_padded_rate = y_min_rate - 0.05 * y_range_rate
                y_max_padded_rate = y_max_rate + 0.05 * y_range_rate
                ax.set_ylim(y_min_padded_rate, y_max_padded_rate)
                yticks_rate = np.linspace(y_min_padded_rate, y_max_padded_rate, 6)
                ax.set_yticks(yticks_rate)
                yticklabels_rate = [f'{rate:.3f} ({-rate*16:.2f})' for rate in yticks_rate]
                ax.set_yticklabels(yticklabels_rate)

                plt.tight_layout()
                plt.savefig(vis_dir / 'affinity_change_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f'Trajectory visualizations saved to: {vis_dir}')
                logger.info(f'  - affinity_trajectories_separate.png')
                logger.info(f'  - selectivity_trajectory.png')
                logger.info(f'  - affinity_trajectories_comparison.png')
                logger.info(f'  - affinity_change_analysis.png')

            except Exception as e:
                logger.warning(f'Failed to generate trajectory visualizations: {e}')
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()