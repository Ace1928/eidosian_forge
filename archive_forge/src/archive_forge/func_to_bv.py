from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def to_bv(self):
    return _to_expr_ref(Z3_mk_char_to_bv(self.ctx_ref(), self.as_ast()), self.ctx)