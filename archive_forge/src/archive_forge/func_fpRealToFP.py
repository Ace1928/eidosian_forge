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
def fpRealToFP(rm, v, sort, ctx=None):
    """Create a Z3 floating-point conversion expression that represents the
    conversion from a real term to a floating-point term.

    >>> x_r = RealVal(1.5)
    >>> x_fp = fpRealToFP(RNE(), x_r, Float32())
    >>> x_fp
    fpToFP(RNE(), 3/2)
    >>> simplify(x_fp)
    1.5
    """
    _z3_assert(is_fprm(rm), 'First argument must be a Z3 floating-point rounding mode expression.')
    _z3_assert(is_real(v), 'Second argument must be a Z3 expression or real sort.')
    _z3_assert(is_fp_sort(sort), 'Third argument must be a Z3 floating-point sort.')
    ctx = _get_ctx(ctx)
    return FPRef(Z3_mk_fpa_to_fp_real(ctx.ref(), rm.ast, v.ast, sort.ast), ctx)