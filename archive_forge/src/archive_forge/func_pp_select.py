import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_select(self, a, d, xs):
    if a.num_args() != 2:
        return self.pp_prefix(a, d, xs)
    else:
        arg1_pp = self.pp_expr(a.arg(0), d + 1, xs)
        arg2_pp = self.pp_expr(a.arg(1), d + 1, xs)
        return compose(arg1_pp, indent(2, compose(to_format('['), arg2_pp, to_format(']'))))