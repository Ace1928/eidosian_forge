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
def import_model_converter(self, other):
    """Import model converter from other into the current solver"""
    Z3_solver_import_model_converter(self.ctx.ref(), other.solver, self.solver)