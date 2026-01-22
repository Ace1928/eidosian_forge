import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_extract(self, a, d, xs):
    high = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 0)
    low = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 1)
    arg = self.pp_expr(a.arg(0), d + 1, xs)
    return seq1(self.pp_name(a), [to_format(high), to_format(low), arg])