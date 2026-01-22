import math
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges
from .ir import LoopBody
from .utils import dominated_nodes
def range_expressable_in_32_bits(range):
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(range.upper)