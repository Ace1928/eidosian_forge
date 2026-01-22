import functools
from typing import Dict, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from ..pattern_matcher import fwd_only, register_replacement
def randperm_index_pattern(x, slice_shape):
    index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
    return (torch.ops.aten.index(x, (index,)), index)