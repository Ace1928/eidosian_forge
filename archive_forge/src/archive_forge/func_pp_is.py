import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_is(self, a, d, xs):
    f = a.params()[0]
    return self.pp_fdecl(f, a, d, xs)