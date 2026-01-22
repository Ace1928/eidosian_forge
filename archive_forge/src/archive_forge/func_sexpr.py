from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
def sexpr(self):
    return Z3_ast_to_string(self.ctx_ref(), self.as_ast())