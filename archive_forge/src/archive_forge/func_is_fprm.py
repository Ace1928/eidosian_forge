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
def is_fprm(a):
    """Return `True` if `a` is a Z3 floating-point rounding mode expression.

    >>> rm = RNE()
    >>> is_fprm(rm)
    True
    >>> rm = 1.0
    >>> is_fprm(rm)
    False
    """
    return isinstance(a, FPRMRef)