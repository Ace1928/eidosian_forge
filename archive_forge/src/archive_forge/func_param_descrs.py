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
def param_descrs(self):
    """Return the parameter description set."""
    return ParamDescrsRef(Z3_tactic_get_param_descrs(self.ctx.ref(), self.tactic), self.ctx)