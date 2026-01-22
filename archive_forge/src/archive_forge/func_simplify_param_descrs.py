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
def simplify_param_descrs():
    """Return the set of parameter descriptions for Z3 `simplify` procedure."""
    return ParamDescrsRef(Z3_simplify_get_param_descrs(main_ctx().ref()), main_ctx())