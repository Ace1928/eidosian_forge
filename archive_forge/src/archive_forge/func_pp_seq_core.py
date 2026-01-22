import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_seq_core(self, f, a, d, xs):
    self.visited = self.visited + 1
    if d > self.max_depth or self.visited > self.max_visited:
        return self.pp_ellipses()
    r = []
    sz = 0
    for elem in a:
        r.append(f(elem, d + 1, xs))
        sz = sz + 1
        if sz > self.max_args:
            r.append(self.pp_ellipses())
            break
    return seq3(r, '[', ']')