import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_fprm_value(self, a):
    _z3_assert(z3.is_fprm_value(a), 'expected FPRMNumRef')
    if self.fpa_pretty and a.decl().kind() in _z3_op_to_fpa_pretty_str:
        return to_format(_z3_op_to_fpa_pretty_str.get(a.decl().kind()))
    else:
        return to_format(_z3_op_to_fpa_normal_str.get(a.decl().kind()))