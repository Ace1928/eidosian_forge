import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def get_tensor_key(obj):
    assert not (obj.dtype.is_floating_point or obj.dtype.is_complex), obj.dtype
    return (obj.data_ptr(), obj.storage_offset(), obj.shape, obj.stride(), obj.dtype)