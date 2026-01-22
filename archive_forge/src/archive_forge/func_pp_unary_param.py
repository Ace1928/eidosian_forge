import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_unary_param(self, a, d, xs):
    p = Z3_get_decl_int_parameter(a.ctx_ref(), a.decl().ast, 0)
    arg = self.pp_expr(a.arg(0), d + 1, xs)
    return seq1(self.pp_name(a), [to_format(p), arg])