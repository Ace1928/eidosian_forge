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
def get_num_levels(self, predicate):
    """Retrieve number of levels used for predicate in PDR engine"""
    return Z3_fixedpoint_get_num_levels(self.ctx.ref(), self.fixedpoint, predicate.ast)