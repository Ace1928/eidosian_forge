from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
def set_allow_smaller_batches(self, is_allow_smaller_batches):
    self.allow_smaller_batches = is_allow_smaller_batches