import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_decl(self, f):
    k = f.kind()
    if k == Z3_OP_DT_IS or k == Z3_OP_ARRAY_MAP:
        g = f.params()[0]
        r = [to_format(g.name())]
        return seq1(self.pp_name(f), r)
    return self.pp_name(f)