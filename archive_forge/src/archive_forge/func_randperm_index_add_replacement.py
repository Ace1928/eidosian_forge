import functools
from typing import Dict, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from ..pattern_matcher import fwd_only, register_replacement
def randperm_index_add_replacement(x, y):
    index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
    return (torch.ops.aten._unsafe_index_put(x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False), index)