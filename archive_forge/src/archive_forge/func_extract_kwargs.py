import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def extract_kwargs(arg):
    kwargs = {'offsets': arg.offsets(), '_max_seqlen': arg._max_seqlen, '_min_seqlen': arg._min_seqlen}
    return kwargs