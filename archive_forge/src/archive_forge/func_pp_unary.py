import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_unary(self, a, d, xs):
    k = a.decl().kind()
    p = self.get_precedence(k)
    child = a.children()[0]
    child_k = None
    if z3.is_app(child):
        child_k = child.decl().kind()
    child_pp = self.pp_expr(child, d + 1, xs)
    if k != child_k and self.is_infix_unary(child_k):
        child_p = self.get_precedence(child_k)
        if p <= child_p:
            child_pp = self.add_paren(child_pp)
    if z3.is_quantifier(child):
        child_pp = self.add_paren(child_pp)
    name = self.pp_name(a)
    return compose(to_format(name), indent(_len(name), child_pp))