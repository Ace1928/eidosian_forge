import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
    return t.broadcast_to(batch_dims + invariant_dims).flatten(0, len(batch_dims) - 1)