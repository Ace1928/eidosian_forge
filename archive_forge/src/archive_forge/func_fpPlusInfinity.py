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
def fpPlusInfinity(s):
    """Create a Z3 floating-point +oo term.

    >>> s = FPSort(8, 24)
    >>> pb = get_fpa_pretty()
    >>> set_fpa_pretty(True)
    >>> fpPlusInfinity(s)
    +oo
    >>> set_fpa_pretty(False)
    >>> fpPlusInfinity(s)
    fpPlusInfinity(FPSort(8, 24))
    >>> set_fpa_pretty(pb)
    """
    _z3_assert(isinstance(s, FPSortRef), 'sort mismatch')
    return FPNumRef(Z3_mk_fpa_inf(s.ctx_ref(), s.ast, False), s.ctx)