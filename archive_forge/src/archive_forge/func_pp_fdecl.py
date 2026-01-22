import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_fdecl(self, f, a, d, xs):
    r = []
    sz = 0
    r.append(to_format(f.name()))
    for child in a.children():
        r.append(self.pp_expr(child, d + 1, xs))
        sz = sz + 1
        if sz > self.max_args:
            r.append(self.pp_ellipses())
            break
    return seq1(self.pp_name(a), r)