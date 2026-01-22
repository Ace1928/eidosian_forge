import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_char(self, a):
    n = a.params()[0]
    return to_format(str(n))