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
def significand_as_bv(self):
    return BitVecNumRef(Z3_fpa_get_numeral_significand_bv(self.ctx.ref(), self.as_ast()), self.ctx)