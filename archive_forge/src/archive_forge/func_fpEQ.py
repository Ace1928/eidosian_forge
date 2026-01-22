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
def fpEQ(a, b, ctx=None):
    """Create the Z3 floating-point expression `fpEQ(other, self)`.

    >>> x, y = FPs('x y', FPSort(8, 24))
    >>> fpEQ(x, y)
    fpEQ(x, y)
    >>> fpEQ(x, y).sexpr()
    '(fp.eq x y)'
    """
    return _mk_fp_bin_pred(Z3_mk_fpa_eq, a, b, ctx)