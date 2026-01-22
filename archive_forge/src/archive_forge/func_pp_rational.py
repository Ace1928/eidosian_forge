import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_rational(self, a):
    if not self.rational_to_decimal:
        return to_format(a.as_string())
    else:
        return to_format(a.as_decimal(self.precision))