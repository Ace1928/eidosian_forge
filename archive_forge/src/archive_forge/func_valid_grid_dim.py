import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def valid_grid_dim(g, mg):
    if g is None:
        return mg
    else:
        return max(1, min(g, mg))