#!/usr/bin/env python3
"""
Evaluate Binding Affinity prediction robustness to diffusion noise
Compares 3 models:
- Model 1: Original KGDiff (joint model)
- Model 2: Head2 1p_all_attention (dual-head sam_pl, Head2 only)
- Model 3: Head2 atom_attention (dual-head sam_pl, Head2 only)

Evaluates at timesteps: 0, 200, 400, 600, 800
- timestep 0: clean ligand (no noise)
- higher timestep: more noise on ligand position and atom types

Noise is applied using the same diffusion schedule as training.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import pickle
import lmdb
import importlib.util
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(os.path.abspath('./'))

import utils.transforms as trans
from datasets.pl_data import ProteinLigandData
from utils.transforms import get_atomic_number_from_index, is_aromatic_from_index


def load_model_from_checkpoint_dir(ckpt_path, device, protein_featurizer, ligand_featurizer, force_use_original=False):
    """
    Load model using the exact molopt_score_model.py from checkpoint directory
    This ensures architecture compatibility with the checkpoint

    Args:
        force_use_original: If True, always use models.molopt_score_model_original (for original KGDiff)
    """
    print(f"Loading checkpoint: {ckpt_path}")

    # Load checkpoint first to check config
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get the ligand atom mode from checkpoint config
    ligand_atom_mode = getattr(ckpt['config'].data.transform, 'ligand_atom_mode', 'add_aromatic')
    print(f"  -> Using ligand atom mode: {ligand_atom_mode}")
    corrected_ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    # For original KGDiff, always use current codebase's original model
    if force_use_original:
        print(f"  -> Force using current codebase's original model")
        from models.molopt_score_model_original import ScorePosNet3D as OriginalScorePosNet3D
        from models.molopt_score_model_original import index_to_log_onehot

        model = OriginalScorePosNet3D(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=corrected_ligand_featurizer.feature_dim
        ).to(device)

        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()

        # Create a mock module for index_to_log_onehot
        import types
        checkpoint_model = types.ModuleType('checkpoint_model')
        checkpoint_model.index_to_log_onehot = index_to_log_onehot

        return model, ckpt['config'], corrected_ligand_featurizer, False, checkpoint_model

    # Get checkpoint directory (remove /checkpoints/xxx.pt)
    ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
    model_file = os.path.join(ckpt_dir, 'models', 'molopt_score_model.py')

    # For original KGDiff, use current codebase model
    if not os.path.exists(model_file):
        print(f"  -> Model file not found in checkpoint dir, using current codebase model")
        from models.molopt_score_model_original import ScorePosNet3D as OriginalScorePosNet3D
        from models.molopt_score_model_original import index_to_log_onehot

        model = OriginalScorePosNet3D(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=corrected_ligand_featurizer.feature_dim
        ).to(device)

        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()

        # Create a mock module for index_to_log_onehot
        import types
        checkpoint_model = types.ModuleType('checkpoint_model')
        checkpoint_model.index_to_log_onehot = index_to_log_onehot

        return model, ckpt['config'], corrected_ligand_featurizer, False, checkpoint_model

    print(f"  -> Loading model architecture from: {model_file}")

    # Dynamically import the model module from checkpoint directory
    spec = importlib.util.spec_from_file_location("checkpoint_model", model_file)
    checkpoint_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_model)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get the ligand atom mode from checkpoint config
    ligand_atom_mode = getattr(ckpt['config'].data.transform, 'ligand_atom_mode', 'add_aromatic')
    print(f"  -> Using ligand atom mode: {ligand_atom_mode}")

    # Create appropriate featurizer based on config
    corrected_ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    # Check if this is sam_pl model
    use_sam_pl = getattr(ckpt['config'].model, 'use_dual_head_sam_pl', False)
    print(f"  -> Training method: {'sam_pl' if use_sam_pl else 'joint'}")

    # Create model using the checkpoint's architecture
    model = checkpoint_model.ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=corrected_ligand_featurizer.feature_dim
    ).to(device)

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)

    if missing_keys:
        print(f"  -> Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"  -> Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

    model.eval()

    return model, ckpt['config'], corrected_ligand_featurizer, use_sam_pl, checkpoint_model


def load_affinity_info():
    """Load affinity information from pickle file"""
    affinity_path = './scratch2/data/affinity_info_complete.pkl'

    if os.path.exists(affinity_path):
        with open(affinity_path, 'rb') as f:
            affinity_info = pickle.load(f)
        return affinity_info
    else:
        raise FileNotFoundError(f"Affinity info file not found: {affinity_path}")


def inject_affinity_to_lmdb(lmdb_path, key, ligand_filename, affinity_info):
    """
    Inject affinity value into LMDB entry
    Based on pl_pair_dataset.py's inject_affinity method
    """
    # Open LMDB in write mode
    db = lmdb.open(lmdb_path, readonly=False, lock=False, subdir=False, map_size=10*(1024**3))

    with db.begin(write=True) as txn:
        value = txn.get(key)
        if value:
            data = pickle.loads(value)

            # Extract ligand filename without extension
            ligand_raw_fn = ligand_filename
            if ligand_raw_fn.endswith('.sdf'):
                ligand_raw_fn = ligand_raw_fn[:-4]

            if ligand_raw_fn in affinity_info:
                # Inject affinity value (use vina score)
                data['affinity'] = affinity_info[ligand_raw_fn]['vina']
                # Update LMDB
                txn.put(key, pickle.dumps(data))
            else:
                print(f"  [Warning] No affinity info for {ligand_raw_fn}")

    db.close()


# Global cache for affinity info (load once)
_AFFINITY_INFO_CACHE = None


def load_sample_from_lmdb(lmdb_path, validation_id):
    """
    Load a single sample from LMDB using sequential index (like sample_diffusion.py)

    Args:
        lmdb_path: Path to LMDB database
        validation_id: Sequential index (0-99 for 100 test samples)

    Returns:
        dict: Sample data
    """
    global _AFFINITY_INFO_CACHE

    # Use CrossDock2020 split file
    split_file = './scratch2/data/crossdocked_pocket10_pose_split.pt'
    split_data = torch.load(split_file, map_location='cpu')
    val_indices = split_data['test']  # CrossDock2020 format: dict with 'train', 'val', 'test'

    if validation_id >= len(val_indices):
        raise ValueError(f"validation_id {validation_id} out of range (max: {len(val_indices)-1})")

    db = lmdb.open(lmdb_path, readonly=True, lock=False, create=False, max_readers=1, subdir=False)

    # Get all LMDB keys first (like sample_diffusion.py)
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

        # Map validation_id to actual test set sample
        # Find the key that corresponds to this test index
        target_idx = val_indices[validation_id]

        # Search for this index in LMDB
        key = str(target_idx).encode()
        value = txn.get(key)

        if value is None:
            # If direct lookup fails, try sequential approach
            # This handles cases where LMDB uses sequential indexing
            if validation_id < len(keys):
                key = keys[validation_id]
                value = txn.get(key)

            if value is None:
                raise ValueError(f"Sample not found for validation_id {validation_id} (target_idx: {target_idx})")

        data = pickle.loads(value)

    db.close()

    # Check if affinity is missing
    if 'affinity' not in data:
        # Load affinity info if not cached
        if _AFFINITY_INFO_CACHE is None:
            print("  [Info] Loading affinity info...")
            _AFFINITY_INFO_CACHE = load_affinity_info()

        # Get ligand filename
        ligand_filename = data.get('ligand_filename', '')

        if ligand_filename:
            # Inject affinity
            inject_affinity_to_lmdb(lmdb_path, key, ligand_filename, _AFFINITY_INFO_CACHE)

            # Reload data
            db = lmdb.open(lmdb_path, readonly=True, lock=False, create=False, max_readers=1, subdir=False)
            with db.begin() as txn:
                value = txn.get(key)
                data = pickle.loads(value)
            db.close()

    return data


def add_diffusion_noise_to_ligand(model, ligand_pos, ligand_v, batch_ligand, timestep, device, model_module, sample_idx=None):
    """
    Add diffusion noise to ligand position and atom types using the SAME method as training

    Args:
        model: trained model (contains alphas_cumprod, q_v_sample)
        ligand_pos: ligand positions [N, 3]
        ligand_v: ligand atom type features [N] (indices)
        batch_ligand: batch assignment [N]
        timestep: diffusion timestep (0-1000)
        device: cuda or cpu
        model_module: imported model module (for index_to_log_onehot function)
        sample_idx: sample index (used for reproducible noise)

    Returns:
        ligand_pos_noisy: noised positions
        ligand_v_noisy: noised atom type indices
    """
    if timestep == 0:
        # No noise at timestep 0
        return ligand_pos, ligand_v

    # Set random seed based on sample_idx and timestep for reproducible noise
    # This ensures all models get the same noise for the same sample and timestep
    if sample_idx is not None:
        seed = sample_idx * 10000 + timestep
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Convert timestep to tensor
    time_step = torch.tensor([timestep], dtype=torch.long, device=device)

    # Get alpha from cumulative product (same as training)
    a = model.alphas_cumprod.index_select(0, time_step)  # (1,)

    # === 1. Position noise (coordinates) ===
    a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
    pos_noise = torch.zeros_like(ligand_pos)
    pos_noise.normal_()
    # Apply noise: Xt = sqrt(a) * X0 + sqrt(1-a) * eps
    ligand_pos_noisy = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

    # === 2. Atom type noise ===
    # Convert indices to log one-hot
    log_ligand_v0 = model_module.index_to_log_onehot(ligand_v, model.num_classes)
    # Sample noisy atom types using model's q_v_sample
    ligand_v_noisy, log_ligand_vt = model.q_v_sample(log_ligand_v0, time_step, batch_ligand)

    return ligand_pos_noisy, ligand_v_noisy


def predict_affinity_single_sample(model, transform, data_raw, device, timestep, use_sam_pl, model_module, use_head2_only=False,
                                   print_ligand_info=False, sample_idx=None):
    """
    Predict binding affinity for a single sample WITH LIGAND DIFFUSION NOISE

    Args:
        timestep: diffusion timestep (0 = no noise, higher = more noise)
        model_module: imported model module (for index_to_log_onehot)
        use_head2_only: If True, return only Head2 prediction (for dual-head models)
        print_ligand_info: If True, print detailed ligand information (coordinates and atom types)
        sample_idx: Sample index for printing

    Returns:
        prediction: Affinity prediction
        ground_truth: Ground truth affinity
        ligand_info_dict: Dictionary containing ligand information (if print_ligand_info=True)
    """
    try:
        # Create a minimal data object for transforms
        temp_data = ProteinLigandData()
        temp_data.protein_pos = data_raw['protein_pos']
        temp_data.protein_element = data_raw['protein_element']
        temp_data.protein_is_backbone = data_raw['protein_is_backbone']
        temp_data.protein_atom_to_aa_type = data_raw['protein_atom_to_aa_type']
        temp_data.ligand_pos = data_raw['ligand_pos']
        temp_data.ligand_element = data_raw['ligand_element']
        temp_data.ligand_bond_index = data_raw['ligand_bond_index']
        temp_data.ligand_bond_type = data_raw['ligand_bond_type']
        temp_data.ligand_atom_feature = data_raw.get('ligand_atom_feature', None)
        temp_data.ligand_hybridization = data_raw.get('ligand_hybridization', None)

        # Add affinity field for NormalizeVina transform
        if 'affinity' not in data_raw:
            raise ValueError("affinity field not found in data")
        temp_data.affinity = torch.tensor(data_raw['affinity'], dtype=torch.float32)

        # Apply transforms
        temp_data = transform(temp_data)

        # Move to device
        temp_data = temp_data.to(device)

        # Create batch tensors (single sample)
        batch_ligand = torch.zeros(len(temp_data.ligand_pos), dtype=torch.long, device=device)
        batch_protein = torch.zeros(len(temp_data.protein_pos), dtype=torch.long, device=device)

        # Store original ligand data for comparison
        ligand_pos_original = temp_data.ligand_pos.clone()
        ligand_v_original = temp_data.ligand_atom_feature_full.clone()

        # === ADD DIFFUSION NOISE TO LIGAND ===
        if timestep > 0:
            ligand_pos_input, ligand_v_input = add_diffusion_noise_to_ligand(
                model, temp_data.ligand_pos, temp_data.ligand_atom_feature_full,
                batch_ligand, timestep, device, model_module, sample_idx=sample_idx
            )
        else:
            # No noise at timestep 0
            ligand_pos_input = temp_data.ligand_pos
            ligand_v_input = temp_data.ligand_atom_feature_full

        # Model prediction with specified timestep
        with torch.no_grad():
            time_step_tensor = torch.tensor([timestep], dtype=torch.long, device=device)

            if use_sam_pl:
                # Sam-PL model: single protein, uses forward_sam_pl method
                preds = model.forward_sam_pl(
                    protein_pos=temp_data.protein_pos,
                    protein_v=temp_data.protein_atom_feature.float(),
                    batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos_input,  # Noisy ligand
                    init_ligand_v=ligand_v_input,      # Noisy ligand
                    batch_ligand=batch_ligand,
                    time_step=time_step_tensor  # Pass actual timestep to model
                )
            else:
                # Regular forward (for original KGDiff - no protein_id parameter)
                preds = model(
                    protein_pos=temp_data.protein_pos,
                    protein_v=temp_data.protein_atom_feature.float(),
                    batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos_input,  # Noisy ligand
                    init_ligand_v=ligand_v_input,      # Noisy ligand
                    batch_ligand=batch_ligand,
                    time_step=time_step_tensor  # Pass actual timestep to model
                )

        # Collect and print detailed ligand information if requested
        ligand_info_dict = None
        if print_ligand_info and sample_idx is not None:
            print(f"\n{'='*100}")
            print(f"Sample {sample_idx} - Timestep {timestep} - Ligand Information")
            print(f"{'='*100}")

            # Get RefinNet predictions from model output
            ligand_pos_pred = None
            ligand_v_pred = None

            # Check for different possible key names (different model versions use different keys)
            # molopt_score_model.py uses: 'ligand_pos', 'ligand_v'
            # molopt_score_model_original.py uses: 'pred_ligand_pos', 'pred_ligand_v'
            if 'ligand_pos' in preds:
                ligand_pos_pred = preds['ligand_pos']
            elif 'pred_ligand_pos' in preds:
                ligand_pos_pred = preds['pred_ligand_pos']
            elif 'ligand_pos_pred' in preds:
                ligand_pos_pred = preds['ligand_pos_pred']

            if 'ligand_v' in preds:
                ligand_v_pred = preds['ligand_v']
            elif 'pred_ligand_v' in preds:
                ligand_v_pred = preds['pred_ligand_v']
            elif 'ligand_v_pred' in preds:
                ligand_v_pred = preds['ligand_v_pred']

            # Initialize ligand info dictionary
            ligand_info_dict = {
                'sample_idx': sample_idx,
                'timestep': timestep,
                'num_atoms': len(ligand_pos_original),
                'original': {
                    'positions': ligand_pos_original.cpu().numpy(),
                    'atom_types': ligand_v_original.cpu().numpy()
                },
                'noisy': {
                    'positions': ligand_pos_input.cpu().numpy(),
                    'atom_types': ligand_v_input.cpu().numpy()
                }
            }

            print(f"\n[Original Ligand]")
            print(f"  Number of atoms: {len(ligand_pos_original)}")
            print(f"  Coordinates (first 5 atoms):")
            for i in range(min(5, len(ligand_pos_original))):
                print(f"    Atom {i}: pos={ligand_pos_original[i].cpu().numpy()}, type_idx={ligand_v_original[i].item()}")

            print(f"\n[Noisy Ligand (Input to RefinNet)]")
            print(f"  Coordinates (first 5 atoms):")
            for i in range(min(5, len(ligand_pos_input))):
                print(f"    Atom {i}: pos={ligand_pos_input[i].cpu().numpy()}, type_idx={ligand_v_input[i].item()}")

            # Calculate position difference (RMSD)
            if timestep > 0:
                pos_diff = torch.sqrt(((ligand_pos_original - ligand_pos_input) ** 2).sum(dim=-1))
                pos_rmsd_mean = pos_diff.mean().item()
                pos_rmsd_max = pos_diff.max().item()
                print(f"\n  Position RMSD from original: {pos_rmsd_mean:.4f} Å (mean), {pos_rmsd_max:.4f} Å (max)")

                # Atom type changes
                type_changes = (ligand_v_original != ligand_v_input).sum().item()
                type_changes_pct = 100*type_changes/len(ligand_v_original)
                print(f"  Atom type changes: {type_changes}/{len(ligand_v_original)} atoms ({type_changes_pct:.1f}%)")

                ligand_info_dict['noise_metrics'] = {
                    'position_rmsd_mean': pos_rmsd_mean,
                    'position_rmsd_max': pos_rmsd_max,
                    'atom_type_changes': type_changes,
                    'atom_type_changes_pct': type_changes_pct
                }

            if ligand_pos_pred is not None:
                print(f"\n[RefinNet Predicted Ligand]")
                print(f"  Coordinates (first 5 atoms):")

                # Convert pred to indices if needed
                if ligand_v_pred is not None:
                    if ligand_v_pred.dim() == 2:
                        ligand_v_pred_idx = ligand_v_pred.argmax(dim=-1)
                    else:
                        ligand_v_pred_idx = ligand_v_pred
                else:
                    ligand_v_pred_idx = None

                for i in range(min(5, len(ligand_pos_pred))):
                    # Handle different tensor formats for atom type prediction
                    if ligand_v_pred_idx is not None:
                        v_idx = ligand_v_pred_idx[i].item()
                    else:
                        v_idx = -1
                    print(f"    Atom {i}: pos={ligand_pos_pred[i].cpu().numpy()}, type_idx={v_idx}")

                # Calculate reconstruction error
                pos_recon_error = torch.sqrt(((ligand_pos_original - ligand_pos_pred) ** 2).sum(dim=-1))
                pos_recon_rmsd_mean = pos_recon_error.mean().item()
                pos_recon_rmsd_max = pos_recon_error.max().item()
                print(f"\n  Position reconstruction RMSD: {pos_recon_rmsd_mean:.4f} Å (mean), {pos_recon_rmsd_max:.4f} Å (max)")

                ligand_info_dict['predicted'] = {
                    'positions': ligand_pos_pred.cpu().numpy(),
                    'atom_types': ligand_v_pred_idx.cpu().numpy() if ligand_v_pred_idx is not None else None
                }
                ligand_info_dict['reconstruction_metrics'] = {
                    'position_rmsd_mean': pos_recon_rmsd_mean,
                    'position_rmsd_max': pos_recon_rmsd_max
                }

                if ligand_v_pred_idx is not None:
                    type_recon_errors = (ligand_v_original != ligand_v_pred_idx).sum().item()
                    type_recon_errors_pct = 100*type_recon_errors/len(ligand_v_original)
                    print(f"  Atom type reconstruction error: {type_recon_errors}/{len(ligand_v_original)} atoms ({type_recon_errors_pct:.1f}%)")
                    ligand_info_dict['reconstruction_metrics']['atom_type_errors'] = type_recon_errors
                    ligand_info_dict['reconstruction_metrics']['atom_type_errors_pct'] = type_recon_errors_pct
            else:
                print(f"\n[RefinNet Predicted Ligand]")
                print(f"  WARNING: No ligand_pos_pred or ligand_v_pred found in model output!")
                print(f"  Available keys in preds: {list(preds.keys())}")
                ligand_info_dict['predicted'] = None

            print(f"{'='*100}\n")

        # Get prediction
        prediction = None

        if use_head2_only:
            # For dual-head models, use only Head2 prediction
            if 'v_cross_attn_pred' in preds:
                # New architecture (atom_attention model)
                v_tensor = preds['v_cross_attn_pred']
            elif 'v_non_interaction_pred' in preds:
                # Old architecture (1p_all_attention model)
                v_tensor = preds['v_non_interaction_pred']
            else:
                print(f"Available prediction keys: {list(preds.keys())}")
                return None, None

            if v_tensor.dim() == 0:
                prediction = v_tensor.item()
            else:
                prediction = v_tensor[0].item() if len(v_tensor) > 0 else None
        else:
            # For original KGDiff (joint model)
            if 'final_exp_pred' in preds:
                prediction = preds['final_exp_pred'].item()
            else:
                print(f"Available prediction keys: {list(preds.keys())}")
                return None, None, None

        ground_truth = temp_data.affinity.item()

        # Add BA prediction to ligand_info_dict if it was created
        if ligand_info_dict is not None:
            ligand_info_dict['binding_affinity'] = {
                'prediction': prediction,
                'ground_truth': ground_truth
            }

        return prediction, ground_truth, ligand_info_dict

    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def evaluate_model_ba_accuracy_with_noise(model, transform, lmdb_path, device, model_name,
                                          num_samples, timesteps, use_sam_pl, model_module, use_head2_only=False,
                                          print_first_n_samples=3):
    """Evaluate model's BA prediction accuracy at different noise levels

    Args:
        print_first_n_samples: Number of samples to print detailed ligand information for (default: 3)
    """
    model.eval()

    # Load validation indices - CrossDock2020 format
    split_file = './scratch2/data/crossdocked_pocket10_pose_split.pt'
    split_data = torch.load(split_file, map_location='cpu')
    val_indices = split_data['test']  # test set
    val_size = len(val_indices)

    actual_samples = min(num_samples, val_size)

    # Store results for each timestep
    results_by_timestep = {}

    # Store ligand information for saving to txt
    all_ligand_info = []

    # Vina score normalization parameters (from NormalizeVina)
    vina_max = 0
    vina_min = -16

    for timestep in timesteps:
        print(f"\n  Evaluating {model_name} at timestep {timestep}...")

        predictions = []
        ground_truths = []
        predictions_vina = []  # Denormalized predictions
        ground_truths_vina = []  # Denormalized ground truths

        # Reconstruction metrics lists
        reconstruction_rmsd_list = []
        reconstruction_type_accuracy_list = []
        ba_error_list = []  # Binding affinity prediction error (absolute)

        failed_samples = []
        missing_lmdb = []
        missing_affinity = []

        for i in tqdm(range(actual_samples), desc=f"{model_name} (t={timestep})"):
            try:
                data_raw = load_sample_from_lmdb(lmdb_path, i)

                # Check if affinity is missing
                if 'affinity' not in data_raw:
                    missing_affinity.append(i)
                    continue

                # Print ligand info for first N samples at each timestep
                print_info = (i < print_first_n_samples)

                pred, gt, ligand_info = predict_affinity_single_sample(
                    model, transform, data_raw, device, timestep, use_sam_pl, model_module, use_head2_only,
                    print_ligand_info=print_info, sample_idx=i
                )

                # Collect ligand info if available
                if ligand_info is not None:
                    ligand_info['model_name'] = model_name
                    all_ligand_info.append(ligand_info)

                    # Collect reconstruction metrics if available
                    if 'reconstruction_metrics' in ligand_info:
                        recon_metrics = ligand_info['reconstruction_metrics']
                        reconstruction_rmsd_list.append(recon_metrics['position_rmsd_mean'])

                        # Calculate atom type accuracy (1 - error rate)
                        if 'atom_type_errors' in recon_metrics:
                            num_atoms = ligand_info['num_atoms']
                            correct_atoms = num_atoms - recon_metrics['atom_type_errors']
                            type_accuracy = correct_atoms / num_atoms
                            reconstruction_type_accuracy_list.append(type_accuracy)

                if pred is not None and gt is not None:
                    predictions.append(pred)
                    ground_truths.append(gt)

                    # Denormalize to Vina score scale: vina = max - normalized * (max - min)
                    pred_vina = vina_max - pred * (vina_max - vina_min)
                    gt_vina = vina_max - gt * (vina_max - vina_min)

                    # Clip ground truth to valid range [vina_min, vina_max] = [-16, 0]
                    # Model only outputs values in this range, so GT should also be clipped
                    gt_vina = np.clip(gt_vina, vina_min, vina_max)

                    predictions_vina.append(pred_vina)
                    ground_truths_vina.append(gt_vina)

                    # Collect BA prediction error (absolute error in Vina scale)
                    ba_error_list.append(abs(pred_vina - gt_vina))
                else:
                    failed_samples.append(i)

            except ValueError as e:
                if "Sample not found" in str(e):
                    missing_lmdb.append(i)
                else:
                    failed_samples.append(i)
            except Exception as e:
                # Other errors
                failed_samples.append(i)
                continue

        # Calculate metrics on Vina score scale
        if len(predictions_vina) > 1:
            predictions_vina_arr = np.array(predictions_vina)
            ground_truths_vina_arr = np.array(ground_truths_vina)

            # Calculate MSE and RMSE on Vina score scale
            mse = np.mean((predictions_vina_arr - ground_truths_vina_arr) ** 2)
            rmse = np.sqrt(mse)

            # Calculate Pearson correlation (same on both scales)
            pearson_r, pearson_p = stats.pearsonr(predictions_vina_arr, ground_truths_vina_arr)

            # Calculate reconstruction metrics statistics
            recon_stats = {}
            if len(reconstruction_rmsd_list) > 0:
                recon_stats['recon_rmsd_mean'] = np.mean(reconstruction_rmsd_list)
                recon_stats['recon_rmsd_std'] = np.std(reconstruction_rmsd_list)
                recon_stats['recon_rmsd_count'] = len(reconstruction_rmsd_list)

            if len(reconstruction_type_accuracy_list) > 0:
                recon_stats['recon_type_accuracy_mean'] = np.mean(reconstruction_type_accuracy_list)
                recon_stats['recon_type_accuracy_std'] = np.std(reconstruction_type_accuracy_list)
                recon_stats['recon_type_accuracy_count'] = len(reconstruction_type_accuracy_list)

            # Calculate BA error statistics
            if len(ba_error_list) > 0:
                recon_stats['ba_error_mean'] = np.mean(ba_error_list)
                recon_stats['ba_error_std'] = np.std(ba_error_list)

            results_by_timestep[timestep] = {
                'timestep': timestep,
                'num_samples': len(predictions_vina),
                'predictions': predictions_vina,  # Store Vina scores
                'ground_truths': ground_truths_vina,  # Store Vina scores
                'mse': mse,
                'rmse': rmse,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'reconstruction_stats': recon_stats
            }

            print(f"    t={timestep}: RMSE={rmse:.4f}, PCC={pearson_r:.4f}, n={len(predictions_vina)}")
            if 'recon_rmsd_mean' in recon_stats:
                print(f"              Recon RMSD: {recon_stats['recon_rmsd_mean']:.4f}±{recon_stats['recon_rmsd_std']:.4f} Å")
            if 'recon_type_accuracy_mean' in recon_stats:
                print(f"              Type Accuracy: {recon_stats['recon_type_accuracy_mean']*100:.2f}±{recon_stats['recon_type_accuracy_std']*100:.2f}%")
            if 'ba_error_mean' in recon_stats:
                print(f"              BA Error: {recon_stats['ba_error_mean']:.4f}±{recon_stats['ba_error_std']:.4f} kcal/mol")

    return results_by_timestep, all_ligand_info


def plot_results(all_results, timesteps, save_dir):
    """Plot MSE and PCC vs noise level, and scatter plots for all models"""
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # Blue, Orange, Red
    markers = ['o', 's', '^']

    # ========== Figure 1: RMSE and PCC vs Timestep ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Binding Affinity Prediction Robustness to Diffusion Noise (Vina Score Scale)', fontsize=14, fontweight='bold')

    # Plot 1: RMSE vs timestep
    ax1 = axes[0]
    for model_name, color, marker in zip(model_names, colors, markers):
        results = all_results[model_name]
        rmse_values = [results[t]['rmse'] for t in timesteps]
        ax1.plot(timesteps, rmse_values, marker=marker, color=color,
                label=model_name, linewidth=2.5, markersize=8)

    ax1.set_xlabel('Diffusion Timestep (Noise Level)', fontsize=12)
    ax1.set_ylabel('RMSE (kcal/mol)', fontsize=12)
    ax1.set_title('RMSE vs Noise Level', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: PCC vs timestep
    ax2 = axes[1]
    for model_name, color, marker in zip(model_names, colors, markers):
        results = all_results[model_name]
        pcc_values = [results[t]['pearson_r'] for t in timesteps]
        ax2.plot(timesteps, pcc_values, marker=marker, color=color,
                label=model_name, linewidth=2.5, markersize=8)

    ax2.set_xlabel('Diffusion Timestep (Noise Level)', fontsize=12)
    ax2.set_ylabel('PCC (Pearson Correlation)', fontsize=12)
    ax2.set_title('PCC vs Noise Level', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_robustness_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved metrics plot: {os.path.join(save_dir, 'noise_robustness_metrics.png')}")

    # ========== Figure 2: Scatter Plots for Each Timestep ==========
    n_timesteps = len(timesteps)
    n_models = len(model_names)

    fig, axes = plt.subplots(n_timesteps, n_models, figsize=(5*n_models, 5*n_timesteps))
    fig.suptitle('Predicted vs Ground Truth Binding Affinity at Different Noise Levels',
                 fontsize=16, fontweight='bold')

    # Ensure axes is 2D even with single row/column
    if n_timesteps == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    # Calculate global min/max for consistent axis scaling
    all_preds = []
    all_gts = []
    for results in all_results.values():
        for t in timesteps:
            all_preds.extend(results[t]['predictions'])
            all_gts.extend(results[t]['ground_truths'])

    min_val = min(min(all_preds), min(all_gts))
    max_val = max(max(all_preds), max(all_gts))
    padding = (max_val - min_val) * 0.05
    axis_min = min_val - padding
    axis_max = max_val + padding

    for i, timestep in enumerate(timesteps):
        for j, model_name in enumerate(model_names):
            ax = axes[i, j]

            results = all_results[model_name][timestep]
            predictions = np.array(results['predictions'])
            ground_truths = np.array(results['ground_truths'])

            # Scatter plot - SWAPPED AXES: ground_truth on x-axis, prediction on y-axis
            ax.scatter(ground_truths, predictions, alpha=0.5, s=20, color=colors[j])

            # Perfect prediction line
            ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.5, linewidth=1.5, label='Perfect')

            # Add metrics text - WITHOUT n_samples
            rmse = results['rmse']
            pcc = results['pearson_r']
            ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nPCC: {pcc:.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Labels and title - SWAPPED
            if i == n_timesteps - 1:
                ax.set_xlabel('Ground Truth Affinity (kcal/mol)', fontsize=11)
            if j == 0:
                ax.set_ylabel('Predicted Affinity (kcal/mol)', fontsize=11)

            # Title: model name on top row, timestep on left column
            title = ""
            if i == 0:
                title = model_name
            if j == 0:
                title = f"t={timestep}" + ("\n" + title if title else "")
            elif i == 0:
                title = model_name
            if title:
                ax.set_title(title, fontsize=11, fontweight='bold')

            # Set consistent axis limits
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_robustness_scatter.png'), dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot: {os.path.join(save_dir, 'noise_robustness_scatter.png')}")


def save_results_to_csv(all_results, timesteps, save_dir):
    """Save results to CSV files"""
    os.makedirs(save_dir, exist_ok=True)

    # Summary table with all models (including reconstruction metrics)
    summary_data = []
    for model_name in all_results.keys():
        for timestep in timesteps:
            result = all_results[model_name][timestep]
            recon_stats = result.get('reconstruction_stats', {})

            row = {
                'model': model_name,
                'timestep': timestep,
                'num_samples': result['num_samples'],
                'mse': result['mse'],
                'rmse': result['rmse'],
                'pearson_r': result['pearson_r'],
                'pearson_p': result['pearson_p'],
                'recon_rmsd_mean': recon_stats.get('recon_rmsd_mean', np.nan),
                'recon_rmsd_std': recon_stats.get('recon_rmsd_std', np.nan),
                'recon_type_accuracy_mean': recon_stats.get('recon_type_accuracy_mean', np.nan),
                'recon_type_accuracy_std': recon_stats.get('recon_type_accuracy_std', np.nan),
                'ba_error_mean': recon_stats.get('ba_error_mean', np.nan),
                'ba_error_std': recon_stats.get('ba_error_std', np.nan)
            }
            summary_data.append(row)

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(save_dir, 'noise_robustness_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    # Pivot table for easy comparison
    pivot_mse = df.pivot(index='timestep', columns='model', values='mse')
    pivot_pcc = df.pivot(index='timestep', columns='model', values='pearson_r')
    pivot_recon_rmsd = df.pivot(index='timestep', columns='model', values='recon_rmsd_mean')
    pivot_recon_type_acc = df.pivot(index='timestep', columns='model', values='recon_type_accuracy_mean')
    pivot_ba_error = df.pivot(index='timestep', columns='model', values='ba_error_mean')

    pivot_mse.to_csv(os.path.join(save_dir, 'mse_comparison.csv'))
    pivot_pcc.to_csv(os.path.join(save_dir, 'pcc_comparison.csv'))
    pivot_recon_rmsd.to_csv(os.path.join(save_dir, 'recon_rmsd_comparison.csv'))
    pivot_recon_type_acc.to_csv(os.path.join(save_dir, 'recon_type_accuracy_comparison.csv'))
    pivot_ba_error.to_csv(os.path.join(save_dir, 'ba_error_comparison.csv'))
    print(f"Saved comparison CSVs: mse_comparison.csv, pcc_comparison.csv, recon_rmsd_comparison.csv, recon_type_accuracy_comparison.csv, ba_error_comparison.csv")


def save_results_table_txt(all_results, timesteps, save_dir):
    """Save results as formatted text table"""
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())

    # Create text table
    txt_path = os.path.join(save_dir, 'results_table.txt')
    with open(txt_path, 'w') as f:
        # Write header
        f.write("="*120 + "\n")
        f.write("Noise Robustness Evaluation Results\n")
        f.write("="*120 + "\n\n")

        # MSE Table
        f.write("Mean Squared Error (MSE) by Timestep\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                mse = all_results[model_name][timestep]['mse']
                f.write(f"{mse:<35.6f}")
            f.write("\n")

        f.write("\n\n")

        # PCC Table
        f.write("Pearson Correlation Coefficient (PCC) by Timestep\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                pcc = all_results[model_name][timestep]['pearson_r']
                f.write(f"{pcc:<35.6f}")
            f.write("\n")

        f.write("\n\n")

        # RMSE Table
        f.write("Root Mean Squared Error (RMSE) by Timestep\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                rmse = all_results[model_name][timestep]['rmse']
                f.write(f"{rmse:<35.6f}")
            f.write("\n")

        f.write("\n\n")

        # Sample counts
        f.write("Number of Samples by Timestep\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                n_samples = all_results[model_name][timestep]['num_samples']
                f.write(f"{n_samples:<35}")
            f.write("\n")

        f.write("\n\n")

        # === RECONSTRUCTION METRICS SECTION ===
        f.write("="*120 + "\n")
        f.write("RefinNet Reconstruction Metrics\n")
        f.write("="*120 + "\n\n")

        # Coordinate reconstruction RMSD (mean ± std)
        f.write("Coordinate Reconstruction RMSD (Å) - Mean ± Std\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                recon_stats = all_results[model_name][timestep].get('reconstruction_stats', {})
                if 'recon_rmsd_mean' in recon_stats:
                    mean = recon_stats['recon_rmsd_mean']
                    std = recon_stats['recon_rmsd_std']
                    f.write(f"{mean:.4f} ± {std:.4f}".ljust(35))
                else:
                    f.write(f"{'N/A':<35}")
            f.write("\n")

        f.write("\n\n")

        # Atom type reconstruction accuracy (mean ± std)
        f.write("Atom Type Reconstruction Accuracy (%) - Mean ± Std\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                recon_stats = all_results[model_name][timestep].get('reconstruction_stats', {})
                if 'recon_type_accuracy_mean' in recon_stats:
                    mean = recon_stats['recon_type_accuracy_mean'] * 100  # Convert to percentage
                    std = recon_stats['recon_type_accuracy_std'] * 100
                    f.write(f"{mean:.2f} ± {std:.2f}".ljust(35))
                else:
                    f.write(f"{'N/A':<35}")
            f.write("\n")

        f.write("\n\n")

        # BA prediction error (mean ± std)
        f.write("Binding Affinity Prediction Error (kcal/mol) - Mean ± Std\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Timestep':<12}")
        for model_name in model_names:
            f.write(f"{model_name:<35}")
        f.write("\n")
        f.write("-"*120 + "\n")

        for timestep in timesteps:
            f.write(f"{timestep:<12}")
            for model_name in model_names:
                recon_stats = all_results[model_name][timestep].get('reconstruction_stats', {})
                if 'ba_error_mean' in recon_stats:
                    mean = recon_stats['ba_error_mean']
                    std = recon_stats['ba_error_std']
                    f.write(f"{mean:.4f} ± {std:.4f}".ljust(35))
                else:
                    f.write(f"{'N/A':<35}")
            f.write("\n")

        f.write("\n")
        f.write("="*120 + "\n")

    print(f"Saved results table: {txt_path}")


def save_ligand_info_to_txt(all_ligand_info, save_dir):
    """Save detailed ligand information to text file (separate file per model)"""
    if not all_ligand_info:
        print("No ligand information to save")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Group by model
    ligand_info_by_model = {}
    for info in all_ligand_info:
        model_name = info['model_name']
        if model_name not in ligand_info_by_model:
            ligand_info_by_model[model_name] = []
        ligand_info_by_model[model_name].append(info)

    # Save separate file for each model
    for model_name, model_ligand_infos in ligand_info_by_model.items():
        # Create safe filename
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        txt_path = os.path.join(save_dir, f'ligand_detailed_info_{safe_model_name}.txt')

        # Sort by sample_idx first, then by timestep
        model_ligand_infos.sort(key=lambda x: (x['sample_idx'], x['timestep']))

        with open(txt_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write(f"Detailed Ligand Information - {model_name}\n")
            f.write("="*120 + "\n\n")
            f.write("This file contains detailed information about ligand coordinates and atom types\n")
            f.write("at different noise levels (timesteps) for the first few samples.\n\n")
            f.write("Organization: Each sample shows all timesteps in sequence\n")
            f.write("  Sample 0: timestep 0, 200, 400, 600, 800\n")
            f.write("  Sample 1: timestep 0, 200, 400, 600, 800\n")
            f.write("  ...\n\n")
            f.write("For each sample-timestep, the following information is provided:\n")
            f.write("  - Original Ligand: Clean ligand from dataset\n")
            f.write("  - Noisy Ligand: Ligand after adding diffusion noise (input to RefinNet)\n")
            f.write("  - Predicted Ligand: Ligand reconstructed by RefinNet\n")
            f.write("  - Noise Metrics: How much noise was added\n")
            f.write("  - Reconstruction Metrics: How well RefinNet reconstructed the ligand\n")
            f.write("  - Binding Affinity Prediction: Model's BA prediction vs ground truth\n")
            f.write("="*120 + "\n\n")

            # Write ligand info sorted by sample, then timestep
            for ligand_info in model_ligand_infos:
                sample_idx = ligand_info['sample_idx']
                timestep = ligand_info['timestep']
                num_atoms = ligand_info['num_atoms']

                f.write("\n" + "="*120 + "\n")
                f.write(f"Sample: {sample_idx} | Timestep: {timestep}\n")
                f.write("="*120 + "\n\n")

                f.write(f"Number of atoms: {num_atoms}\n\n")

                # Original ligand
                f.write("[Original Ligand]\n")
                f.write("-"*80 + "\n")
                original_pos = ligand_info['original']['positions']
                original_types = ligand_info['original']['atom_types']
                f.write(f"{'Atom':<6} {'X':>10} {'Y':>10} {'Z':>10} {'Type':>8}\n")
                f.write("-"*80 + "\n")
                for i in range(min(10, num_atoms)):  # Show first 10 atoms
                    pos = original_pos[i]
                    atype = original_types[i]
                    f.write(f"{i:<6} {pos[0]:>10.3f} {pos[1]:>10.3f} {pos[2]:>10.3f} {atype:>8}\n")
                if num_atoms > 10:
                    f.write(f"... ({num_atoms - 10} more atoms)\n")
                f.write("\n")

                # Noisy ligand
                f.write("[Noisy Ligand - Input to RefinNet]\n")
                f.write("-"*80 + "\n")
                noisy_pos = ligand_info['noisy']['positions']
                noisy_types = ligand_info['noisy']['atom_types']
                f.write(f"{'Atom':<6} {'X':>10} {'Y':>10} {'Z':>10} {'Type':>8}\n")
                f.write("-"*80 + "\n")
                for i in range(min(10, num_atoms)):
                    pos = noisy_pos[i]
                    atype = noisy_types[i]
                    f.write(f"{i:<6} {pos[0]:>10.3f} {pos[1]:>10.3f} {pos[2]:>10.3f} {atype:>8}\n")
                if num_atoms > 10:
                    f.write(f"... ({num_atoms - 10} more atoms)\n")
                f.write("\n")

                # Noise metrics (if timestep > 0)
                if 'noise_metrics' in ligand_info:
                    noise_metrics = ligand_info['noise_metrics']
                    f.write("[Noise Metrics]\n")
                    f.write("-"*80 + "\n")
                    f.write(f"  Position RMSD from original:\n")
                    f.write(f"    Mean: {noise_metrics['position_rmsd_mean']:.4f} Å\n")
                    f.write(f"    Max:  {noise_metrics['position_rmsd_max']:.4f} Å\n")
                    f.write(f"  Atom type changes: {noise_metrics['atom_type_changes']}/{num_atoms} atoms ")
                    f.write(f"({noise_metrics['atom_type_changes_pct']:.1f}%)\n\n")

                # Predicted ligand
                if ligand_info.get('predicted') is not None:
                    f.write("[RefinNet Predicted Ligand]\n")
                    f.write("-"*80 + "\n")
                    pred_pos = ligand_info['predicted']['positions']
                    pred_types = ligand_info['predicted']['atom_types']
                    f.write(f"{'Atom':<6} {'X':>10} {'Y':>10} {'Z':>10} {'Type':>8}\n")
                    f.write("-"*80 + "\n")
                    for i in range(min(10, num_atoms)):
                        pos = pred_pos[i]
                        atype = pred_types[i] if pred_types is not None else -1
                        f.write(f"{i:<6} {pos[0]:>10.3f} {pos[1]:>10.3f} {pos[2]:>10.3f} {atype:>8}\n")
                    if num_atoms > 10:
                        f.write(f"... ({num_atoms - 10} more atoms)\n")
                    f.write("\n")

                    # Reconstruction metrics
                    if 'reconstruction_metrics' in ligand_info:
                        recon_metrics = ligand_info['reconstruction_metrics']
                        f.write("[Reconstruction Metrics]\n")
                        f.write("-"*80 + "\n")
                        f.write(f"  Position reconstruction RMSD:\n")
                        f.write(f"    Mean: {recon_metrics['position_rmsd_mean']:.4f} Å\n")
                        f.write(f"    Max:  {recon_metrics['position_rmsd_max']:.4f} Å\n")
                        if 'atom_type_errors' in recon_metrics:
                            f.write(f"  Atom type reconstruction error: {recon_metrics['atom_type_errors']}/{num_atoms} atoms ")
                            f.write(f"({recon_metrics['atom_type_errors_pct']:.1f}%)\n")
                        f.write("\n")
                else:
                    f.write("[RefinNet Predicted Ligand]\n")
                    f.write("-"*80 + "\n")
                    f.write("  WARNING: No prediction available from RefinNet\n\n")

                # Binding Affinity Prediction
                if 'binding_affinity' in ligand_info:
                    ba_info = ligand_info['binding_affinity']
                    f.write("[Binding Affinity Prediction]\n")
                    f.write("-"*80 + "\n")

                    # Denormalize values to Vina score scale
                    vina_max = 0
                    vina_min = -16
                    pred_normalized = ba_info['prediction']
                    gt_normalized = ba_info['ground_truth']

                    # Convert to Vina scale: vina = max - normalized * (max - min)
                    pred_vina = vina_max - pred_normalized * (vina_max - vina_min)
                    gt_vina = vina_max - gt_normalized * (vina_max - vina_min)

                    # Calculate error
                    error = abs(pred_vina - gt_vina)

                    f.write(f"  Ground Truth (Vina): {gt_vina:.4f} kcal/mol\n")
                    f.write(f"  Predicted (Vina):    {pred_vina:.4f} kcal/mol\n")
                    f.write(f"  Absolute Error:      {error:.4f} kcal/mol\n")
                    f.write("\n")

            f.write("\n" + "="*120 + "\n")
            f.write("End of detailed ligand information\n")
            f.write("="*120 + "\n")

        print(f"Saved detailed ligand information for {model_name}: {txt_path}")


def save_predicted_ligands_for_pymol(all_ligand_info, save_dir):
    """
    Save predicted ligands as SDF and PDB files for PyMol visualization

    Args:
        all_ligand_info: List of ligand info dictionaries containing predicted ligand data
        save_dir: Directory to save the structure files
    """
    if not all_ligand_info:
        print("No ligand information to save")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Group by model
    ligand_info_by_model = {}
    for info in all_ligand_info:
        model_name = info['model_name']
        if model_name not in ligand_info_by_model:
            ligand_info_by_model[model_name] = []
        ligand_info_by_model[model_name].append(info)

    # Process each model
    for model_name, model_ligand_infos in ligand_info_by_model.items():
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        model_dir = os.path.join(save_dir, f'predicted_ligands_{safe_model_name}')
        os.makedirs(model_dir, exist_ok=True)

        saved_count = 0
        skipped_count = 0

        for ligand_info in model_ligand_infos:
            sample_idx = ligand_info['sample_idx']
            timestep = ligand_info['timestep']
            sample_dir = os.path.join(model_dir, f'sample_{sample_idx:03d}')
            os.makedirs(sample_dir, exist_ok=True)

            # Helper function to save a ligand structure
            def save_ligand_structure(positions, atom_types, suffix):
                if atom_types is None:
                    return False

                try:
                    # Convert atom type indices to atomic numbers
                    atomic_numbers = get_atomic_number_from_index(
                        torch.LongTensor(atom_types), mode='add_aromatic'
                    )
                    is_aromatic_flags = is_aromatic_from_index(
                        torch.LongTensor(atom_types), mode='add_aromatic'
                    )

                    # Create RDKit molecule
                    mol = Chem.RWMol()
                    atom_indices = []

                    for i, (atomic_num, is_arom) in enumerate(zip(atomic_numbers, is_aromatic_flags)):
                        atom = Chem.Atom(int(atomic_num))
                        if is_arom:
                            atom.SetIsAromatic(True)
                        idx = mol.AddAtom(atom)
                        atom_indices.append(idx)

                    # If atom addition failed, return False
                    if len(atom_indices) != len(atomic_numbers):
                        return False

                    # Convert to RDKit molecule
                    mol = mol.GetMol()

                    # Add 3D coordinates
                    conf = Chem.Conformer(len(atomic_numbers))
                    for i, pos in enumerate(positions):
                        conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
                    mol.AddConformer(conf)

                    # Save as SDF
                    sdf_path = os.path.join(sample_dir, f'timestep_{timestep:04d}_{suffix}.sdf')
                    writer = Chem.SDWriter(sdf_path)
                    writer.write(mol)
                    writer.close()

                    # Save as PDB (try sanitizing first to fix aromatic issues)
                    pdb_path = os.path.join(sample_dir, f'timestep_{timestep:04d}_{suffix}.pdb')
                    try:
                        # Try to sanitize the molecule
                        Chem.SanitizeMol(mol)
                        Chem.MolToPDBFile(mol, pdb_path)
                    except:
                        # If sanitization fails, try without aromatic info
                        try:
                            mol_copy = Chem.RWMol()
                            for atom in mol.GetAtoms():
                                new_atom = Chem.Atom(atom.GetAtomicNum())
                                mol_copy.AddAtom(new_atom)
                            mol_copy = mol_copy.GetMol()
                            mol_copy.AddConformer(conf)
                            Chem.MolToPDBFile(mol_copy, pdb_path)
                        except:
                            # If PDB still fails, at least we have SDF
                            pass

                    return True

                except Exception as e:
                    print(f"  Warning: Failed to save {suffix} ligand for sample {sample_idx}, timestep {timestep}: {e}")
                    return False

            # Save original ligand (once per sample, at timestep 0)
            if timestep == 0:
                orig_pos = ligand_info['original']['positions']
                orig_types = ligand_info['original']['atom_types']
                save_ligand_structure(orig_pos, orig_types, 'original')

            # Save noisy ligand (if timestep > 0)
            if timestep > 0:
                noisy_pos = ligand_info['noisy']['positions']
                noisy_types = ligand_info['noisy']['atom_types']
                save_ligand_structure(noisy_pos, noisy_types, 'noisy')

            # Save predicted ligand
            if ligand_info.get('predicted') is not None:
                pred_pos = ligand_info['predicted']['positions']
                pred_types = ligand_info['predicted']['atom_types']
                success = save_ligand_structure(pred_pos, pred_types, 'predicted')
                if success:
                    saved_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

        print(f"Saved {saved_count} predicted ligands for {model_name} (skipped {skipped_count})")
        print(f"  Output directory: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate noise robustness of three models')
    parser.add_argument('--model1_ckpt', type=str,
                      default='./logs_diffusion/training_2023_06_06__20_16_16/checkpoints/642000.pt',
                      help='Original KGDiff checkpoint')
    parser.add_argument('--model2_ckpt', type=str,
                      default='./logs_diffusion_dual_head_sam_pl/training_dual_head_sam_pl_1_all_2025_10_19__21_39_12_dual_head_sam_pl/checkpoints/810000.pt',
                      help='Head2 1p_all_attention checkpoint')
    parser.add_argument('--model3_ckpt', type=str,
                      default='./logs_diffusion_dual_head_sam_pl/training_dual_head_sam_pl_2025_10_15__15_00_00_dual_head_sam_pl/checkpoints/844000.pt',
                      help='Head2 atom_attention checkpoint')
    parser.add_argument('--lmdb_path', type=str,
                      default='./scratch2/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--timesteps', type=str, default='0,200,400,600,800',
                      help='Comma-separated list of diffusion timesteps')
    parser.add_argument('--save_dir', type=str, default='./noise_robustness_evaluation_results')
    parser.add_argument('--print_first_n_samples', type=int, default=3,
                      help='Number of samples to print detailed ligand information for (default: 3)')

    args = parser.parse_args()

    # Parse timesteps
    timesteps = [int(t) for t in args.timesteps.split(',')]

    print(f"\n{'='*80}")
    print("Noise Robustness Evaluation - Binding Affinity Prediction")
    print(f"{'='*80}")
    print(f"Model 1: Original KGDiff (joint)")
    print(f"Model 2: Head2 1p_all_attention (dual-head sam_pl, Head2 only)")
    print(f"Model 3: Head2 atom_attention (dual-head sam_pl, Head2 only)")
    print(f"Timesteps: {timesteps}")
    print(f"(timestep 0 = clean ligand, higher timestep = more noise)\n")

    # Setup transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom('add_aromatic')

    all_results = {}
    all_ligand_info_combined = []  # Collect ligand info from all models

    # List of models to evaluate
    models_to_evaluate = [
        (args.model1_ckpt, 'Original KGDiff', False, True),   # (checkpoint, name, use_head2_only, force_use_original)
        (args.model2_ckpt, 'Head2 1p_all_attention', True, False),
        (args.model3_ckpt, 'Head2 atom_attention', True, False)
    ]

    # Evaluate each model
    for model_ckpt, model_name, use_head2_only, force_use_original in models_to_evaluate:
        if os.path.exists(model_ckpt):
            try:
                print(f"\n{'='*80}")
                print(f"Loading {model_name}: {model_ckpt}")
                print(f"{'='*80}")

                model, config, model_ligand_featurizer, use_sam_pl, model_module = load_model_from_checkpoint_dir(
                    model_ckpt, args.device, protein_featurizer, ligand_featurizer, force_use_original
                )

                # Create transform with correct featurizer
                model_transform = Compose([
                    protein_featurizer,
                    model_ligand_featurizer,
                    trans.FeaturizeLigandBond(),
                    trans.NormalizeVina('crossdock2020')
                ])

                results, ligand_info = evaluate_model_ba_accuracy_with_noise(
                    model, model_transform, args.lmdb_path, args.device,
                    model_name, args.num_samples, timesteps, use_sam_pl, model_module, use_head2_only,
                    print_first_n_samples=args.print_first_n_samples
                )

                if results:
                    all_results[model_name] = results

                # Collect ligand info
                if ligand_info:
                    all_ligand_info_combined.extend(ligand_info)

                # Clean up GPU memory
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error loading/evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Checkpoint not found: {model_ckpt}")

    if not all_results:
        print("Error: No valid models evaluated!")
        return

    # Plot results
    plot_results(all_results, timesteps, args.save_dir)

    # Save results to CSV
    save_results_to_csv(all_results, timesteps, args.save_dir)

    # Save results as formatted text table
    save_results_table_txt(all_results, timesteps, args.save_dir)

    # Save detailed results as pickle
    pickle_path = os.path.join(args.save_dir, 'all_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved detailed results: {pickle_path}")

    # Save detailed ligand information to txt
    save_ligand_info_to_txt(all_ligand_info_combined, args.save_dir)

    # Save ligand info as pickle for later conversion
    ligand_info_pickle_path = os.path.join(args.save_dir, 'all_ligand_info.pkl')
    with open(ligand_info_pickle_path, 'wb') as f:
        pickle.dump(all_ligand_info_combined, f)
    print(f"Saved ligand info pickle: {ligand_info_pickle_path}")

    # Save predicted ligands as SDF and PDB for PyMol visualization
    print(f"\nSaving predicted ligands for PyMol visualization...")
    save_predicted_ligands_for_pymol(all_ligand_info_combined, args.save_dir)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
