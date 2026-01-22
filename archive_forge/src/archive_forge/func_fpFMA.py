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
def fpFMA(rm, a, b, c, ctx=None):
    """Create a Z3 floating-point fused multiply-add expression.
    """
    return _mk_fp_tern(Z3_mk_fpa_fma, rm, a, b, c, ctx)