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
def register_relation(self, *relations):
    """Register relation as recursive"""
    relations = _get_args(relations)
    for f in relations:
        Z3_fixedpoint_register_relation(self.ctx.ref(), self.fixedpoint, f.ast)