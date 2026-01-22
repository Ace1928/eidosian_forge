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
def fpRem(a, b, ctx=None):
    """Create a Z3 floating-point remainder expression.

    >>> s = FPSort(8, 24)
    >>> x = FP('x', s)
    >>> y = FP('y', s)
    >>> fpRem(x, y)
    fpRem(x, y)
    >>> fpRem(x, y).sort()
    FPSort(8, 24)
    """
    return _mk_fp_bin_norm(Z3_mk_fpa_rem, a, b, ctx)