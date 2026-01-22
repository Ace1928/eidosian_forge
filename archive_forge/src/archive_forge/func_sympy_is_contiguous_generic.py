import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def sympy_is_contiguous_generic(sizes, strides, dim_order):
    import sympy
    dim = len(sizes)
    if len(dim_order) != dim:
        return sympy.false
    is_contiguous = sympy.true
    z = sympy.Integer(1)
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.Integer(1)) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.Integer(0))
    return is_contiguous