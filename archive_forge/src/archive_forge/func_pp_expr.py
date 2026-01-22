import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_expr(self, a, d, xs):
    self.visited = self.visited + 1
    if d > self.max_depth or self.visited > self.max_visited:
        return self.pp_ellipses()
    if z3.is_app(a):
        return self.pp_app(a, d, xs)
    elif z3.is_quantifier(a):
        return self.pp_quantifier(a, d, xs)
    elif z3.is_var(a):
        return self.pp_var(a, d, xs)
    else:
        return to_format(self.pp_unknown())