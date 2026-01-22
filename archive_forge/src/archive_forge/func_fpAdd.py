from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def fpAdd(rm, a, b, ctx=None):
    """Create a Z3 floating-point addition expression.

    >>> s = FPSort(8, 24)
    >>> rm = RNE()
    >>> x = FP('x', s)
    >>> y = FP('y', s)
    >>> fpAdd(rm, x, y)
    x + y
    >>> fpAdd(RTZ(), x, y) # default rounding mode is RTZ
    fpAdd(RTZ(), x, y)
    >>> fpAdd(rm, x, y).sort()
    FPSort(8, 24)
    """
    return _mk_fp_bin(Z3_mk_fpa_add, rm, a, b, ctx)