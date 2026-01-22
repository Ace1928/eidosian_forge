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
def set_default_rounding_mode(rm, ctx=None):
    global _dflt_rounding_mode
    if is_fprm_value(rm):
        _dflt_rounding_mode = rm.decl().kind()
    else:
        _z3_assert(_dflt_rounding_mode in _ROUNDING_MODES, 'illegal rounding mode')
        _dflt_rounding_mode = rm