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
def num_no_patterns(self):
    """Return the number of no-patterns."""
    return Z3_get_quantifier_num_no_patterns(self.ctx_ref(), self.ast)