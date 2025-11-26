import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset, PDBPairDataset, MultiProteinPairedDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    
    # Debug: log dataset creation
    with open('/tmp/get_dataset_log.txt', 'w') as f:
        f.write(f"get_dataset called with name={name}, config attributes: {dir(config)}\n")
        f.write(f"config.__dict__: {config.__dict__ if hasattr(config, '__dict__') else 'no __dict__'}\n")
        f.write(f"args: {args}\n")
        f.write(f"kwargs: {kwargs}\n")
    
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pdbbind':
        dataset = PDBPairDataset(root, *args, **kwargs)
    elif name == 'multipro':
        with open('/tmp/get_dataset_log.txt', 'a') as f:
            f.write(f"Creating MultiProteinPairedDataset\n")
        
        dataset = MultiProteinPairedDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
