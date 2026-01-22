import math
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges
from .ir import LoopBody
from .utils import dominated_nodes
def val_expressable_in_32_bits(val):
    if getattr(val, 'is_Boolean', False):
        return True
    if isinstance(val, sympy.Expr):
        assert val.is_number
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)
    if isinstance(val, float):
        return val <= 2 ** 24 and val >= -2 ** 24
    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min
    raise Exception(f'Unexpected value {val}')