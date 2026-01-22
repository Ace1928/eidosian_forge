import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_power(self, a, d, xs):
    arg1_pp = self.pp_power_arg(a.arg(0), d + 1, xs)
    arg2_pp = self.pp_expr(a.arg(1), d + 1, xs)
    return compose(arg1_pp, to_format('<sup>', 1), arg2_pp, to_format('</sup>', 1))