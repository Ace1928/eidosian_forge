import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def infix_args(self, a, d, xs):
    r = []
    self.infix_args_core(a, d, xs, r)
    return r