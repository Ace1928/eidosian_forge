import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_set(self, id, a):
    return seq1(id, [self.pp_sort(a.sort())])