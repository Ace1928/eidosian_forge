import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
def validate_state(self):
    assert self.seed.numel() != 0 and self.base_offset.numel() != 0