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
def fpFPToFP(rm, v, sort, ctx=None):
    """Create a Z3 floating-point conversion expression that represents the
    conversion from a floating-point term to a floating-point term of different precision.

    >>> x_sgl = FPVal(1.0, Float32())
    >>> x_dbl = fpFPToFP(RNE(), x_sgl, Float64())
    >>> x_dbl
    fpToFP(RNE(), 1)
    >>> simplify(x_dbl)
    1
    >>> x_dbl.sort()
    FPSort(11, 53)
    """
    _z3_assert(is_fprm(rm), 'First argument must be a Z3 floating-point rounding mode expression.')
    _z3_assert(is_fp(v), 'Second argument must be a Z3 floating-point expression.')
    _z3_assert(is_fp_sort(sort), 'Third argument must be a Z3 floating-point sort.')
    ctx = _get_ctx(ctx)
    return FPRef(Z3_mk_fpa_to_fp_float(ctx.ref(), rm.ast, v.ast, sort.ast), ctx)