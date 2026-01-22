import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_map(self, a, d, xs):
    f = z3.get_map_func(a)
    return self.pp_fdecl(f, a, d, xs)