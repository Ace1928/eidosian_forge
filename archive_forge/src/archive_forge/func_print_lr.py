import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def print_lr(self, is_verbose, group, lr, epoch=None):
    """Display the current learning rate.
        """
    if is_verbose:
        if epoch is None:
            print(f'Adjusting learning rate of group {group} to {lr:.4e}.')
        else:
            epoch_str = ('%.2f' if isinstance(epoch, float) else '%.5d') % epoch
            print(f'Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}.')