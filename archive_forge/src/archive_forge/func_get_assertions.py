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
def get_assertions(self):
    """retrieve assertions that have been added to fixedpoint context"""
    return AstVector(Z3_fixedpoint_get_assertions(self.ctx.ref(), self.fixedpoint), self.ctx)