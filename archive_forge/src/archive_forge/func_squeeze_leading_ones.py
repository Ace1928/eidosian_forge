import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def squeeze_leading_ones(t):
    while t.shape[0] == 1:
        t = t.squeeze(0)
    return t