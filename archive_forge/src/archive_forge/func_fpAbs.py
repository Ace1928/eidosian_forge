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
def fpAbs(a, ctx=None):
    """Create a Z3 floating-point absolute value expression.

    >>> s = FPSort(8, 24)
    >>> rm = RNE()
    >>> x = FPVal(1.0, s)
    >>> fpAbs(x)
    fpAbs(1)
    >>> y = FPVal(-20.0, s)
    >>> y
    -1.25*(2**4)
    >>> fpAbs(y)
    fpAbs(-1.25*(2**4))
    >>> fpAbs(-1.25*(2**4))
    fpAbs(-1.25*(2**4))
    >>> fpAbs(x).sort()
    FPSort(8, 24)
    """
    ctx = _get_ctx(ctx)
    [a] = _coerce_fp_expr_list([a], ctx)
    return FPRef(Z3_mk_fpa_abs(ctx.ref(), a.as_ast()), ctx)