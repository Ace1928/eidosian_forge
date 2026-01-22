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
def significand_as_long(self):
    ptr = (ctypes.c_ulonglong * 1)()
    if not Z3_fpa_get_numeral_significand_uint64(self.ctx.ref(), self.as_ast(), ptr):
        raise Z3Exception('error retrieving the significand of a numeral.')
    return ptr[0]