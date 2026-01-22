from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
def set_batch_second(ew):
    ew.set_batch_first(False)