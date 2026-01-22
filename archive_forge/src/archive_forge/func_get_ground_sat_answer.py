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
def get_ground_sat_answer(self):
    """Retrieve a ground cex from last query call."""
    r = Z3_fixedpoint_get_ground_sat_answer(self.ctx.ref(), self.fixedpoint)
    return _to_expr_ref(r, self.ctx)