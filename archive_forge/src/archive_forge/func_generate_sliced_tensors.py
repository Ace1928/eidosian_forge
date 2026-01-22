import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def generate_sliced_tensors(slices):
    for t, t_dims in tensor_dims_map.items():
        yield next(multidim_slicer(t_dims, slices, t))