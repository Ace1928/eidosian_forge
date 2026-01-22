import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
@classmethod
def get_updated_bwd_offset(cls):
    if not cls.bwd_state.offset_advanced_alteast_once:
        return cls.bwd_state.base_offset
    return cls.multiple_of_4(cls.bwd_state.base_offset + cls.bwd_state.relative_offset)