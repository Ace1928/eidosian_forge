import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_compose(self, f, indent):
    for c in f.children:
        self.pp(c, indent)