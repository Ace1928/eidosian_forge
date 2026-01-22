from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
def get_sl(self):
    if not self._get_sl_called_within_step:
        warnings.warn('To get the last sparsity level computed by the scheduler, please use `get_last_sl()`.')
    raise NotImplementedError