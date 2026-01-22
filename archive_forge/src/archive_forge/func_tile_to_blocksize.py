import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def tile_to_blocksize(t, blocksize):
    *rest, m, n = t.shape
    new_shape = rest + [m // blocksize[0], blocksize[0], n // blocksize[1], blocksize[1]]
    return t.view(new_shape).transpose(-3, -2)