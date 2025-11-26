"""
Utility function to fix data paths in checkpoint configs
"""

def fix_checkpoint_data_paths(ckpt, logger=None):
    """
    Fix data paths in checkpoint config to use new scratch2/data location

    Args:
        ckpt: Checkpoint dictionary containing 'config'
        logger: Optional logger for info messages

    Returns:
        Modified checkpoint (modifies in-place and returns)
    """
    if not hasattr(ckpt, 'get'):
        return ckpt

    if 'config' not in ckpt:
        return ckpt

    config = ckpt['config']

    if not hasattr(config, 'data'):
        return ckpt

    # Fix data.path
    if hasattr(config.data, 'path'):
        old_path = config.data.path
        updated = False

        # Replace ./data/ with ./scratch2/data/
        if old_path.startswith('./data/'):
            config.data.path = old_path.replace('./data/', './scratch2/data/')
            updated = True
        # Replace /home/ktori1361/KGDiff/data with /home/ktori1361/KGDiff/scratch2/data
        elif '/KGDiff/data/' in old_path:
            config.data.path = old_path.replace('/KGDiff/data/', '/KGDiff/scratch2/data/')
            updated = True

        if updated and logger:
            logger.info(f"Updated data path: {old_path} -> {config.data.path}")

    # Fix data.split
    if hasattr(config.data, 'split'):
        old_split = config.data.split
        updated = False

        # Replace ./data/ with ./scratch2/data/
        if old_split.startswith('./data/'):
            config.data.split = old_split.replace('./data/', './scratch2/data/')
            updated = True
        # Replace /home/ktori1361/KGDiff/data with /home/ktori1361/KGDiff/scratch2/data
        elif '/KGDiff/data/' in old_split:
            config.data.split = old_split.replace('/KGDiff/data/', '/KGDiff/scratch2/data/')
            updated = True

        if updated and logger:
            logger.info(f"Updated split path: {old_split} -> {config.data.split}")

    return ckpt
