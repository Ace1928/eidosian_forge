import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_distinct(self, a, d, xs):
    if a.num_args() == 2:
        op = self.pp_neq()
        sz = _len(op)
        op.string = ' ' + op.string
        op.size = sz + 1
        return group(seq(self.infix_args(a, d, xs), op))
    else:
        return self.pp_prefix(a, d, xs)