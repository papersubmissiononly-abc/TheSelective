import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        # Create ligand neighbor list safely
        if hasattr(instance, 'ligand_bond_index') and instance.ligand_bond_index.numel() > 0:
            instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                      if instance.ligand_bond_index[0, k].item() == i]
                                           for i in instance.ligand_bond_index[0]}
        else:
            instance['ligand_nbh_list'] = {}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            # Check if ligand_element exists and is not empty
            if 'ligand_element' in self and self['ligand_element'].numel() > 0:
                return self['ligand_element'].size(0)
            else:
                return 0
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        # Use safe collate function to handle empty tensors
        kwargs['collate_fn'] = safe_collate_fn
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def safe_collate_fn(batch):
    """Safe collate function that handles empty tensors and lists"""
    # Filter out None values
    batch = [data for data in batch if data is not None]
    
    if len(batch) == 0:
        raise ValueError("Empty batch after filtering None values")
    
    # Check for empty tensors in critical fields
    filtered_batch = []
    for data in batch:
        # Skip data items with empty critical tensors
        if (hasattr(data, 'ligand_element') and data.ligand_element.numel() == 0) or \
           (hasattr(data, 'protein_element') and data.protein_element.numel() == 0):
            continue
        filtered_batch.append(data)
    
    if len(filtered_batch) == 0:
        raise ValueError("No valid data items in batch after filtering empty tensors")
    
    # Use the default PyG collate
    return Batch.from_data_list(filtered_batch, follow_batch=FOLLOW_BATCH)


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int)
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix