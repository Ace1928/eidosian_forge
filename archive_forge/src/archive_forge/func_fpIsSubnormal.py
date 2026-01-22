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
def fpIsSubnormal(a, ctx=None):
    """Create a Z3 floating-point isSubnormal expression.
    """
    return _mk_fp_unary_pred(Z3_mk_fpa_is_subnormal, a, ctx)