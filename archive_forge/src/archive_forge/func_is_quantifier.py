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
def is_quantifier(a):
    """Return `True` if `a` is a Z3 quantifier.

    >>> f = Function('f', IntSort(), IntSort())
    >>> x = Int('x')
    >>> q = ForAll(x, f(x) == 0)
    >>> is_quantifier(q)
    True
    >>> is_quantifier(f(x))
    False
    """
    return isinstance(a, QuantifierRef)