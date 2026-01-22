import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_neq(self):
    return to_format('&ne;')