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
def fpToIEEEBV(x, ctx=None):
    """\x08rief Conversion of a floating-point term into a bit-vector term in IEEE 754-2008 format.

    The size of the resulting bit-vector is automatically determined.

    Note that IEEE 754-2008 allows multiple different representations of NaN. This conversion
    knows only one NaN and it will always produce the same bit-vector representation of
    that NaN.

    >>> x = FP('x', FPSort(8, 24))
    >>> y = fpToIEEEBV(x)
    >>> print(is_fp(x))
    True
    >>> print(is_bv(y))
    True
    >>> print(is_fp(y))
    False
    >>> print(is_bv(x))
    False
    """
    if z3_debug():
        _z3_assert(is_fp(x), 'First argument must be a Z3 floating-point expression')
    ctx = _get_ctx(ctx)
    return BitVecRef(Z3_mk_fpa_to_ieee_bv(ctx.ref(), x.ast), ctx)