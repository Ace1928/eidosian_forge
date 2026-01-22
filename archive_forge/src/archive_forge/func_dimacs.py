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
def dimacs(self, include_names=True):
    """Return a textual representation of the solver in DIMACS format."""
    return Z3_solver_to_dimacs_string(self.ctx.ref(), self.solver, include_names)