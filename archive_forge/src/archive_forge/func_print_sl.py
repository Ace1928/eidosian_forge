from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
def print_sl(self, is_verbose, group, sl, epoch=None):
    """Display the current sparsity level.
        """
    if is_verbose:
        if epoch is None:
            print(f'Adjusting sparsity level of group {group} to {sl:.4e}.')
        else:
            print(f'Epoch {epoch:5d}: adjusting sparsity level of group {group} to {sl:.4e}.')