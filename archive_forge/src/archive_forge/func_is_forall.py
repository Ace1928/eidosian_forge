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
def is_forall(self):
    """Return `True` if `self` is a universal quantifier.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) == 0)
        >>> q.is_forall()
        True
        >>> q = Exists(x, f(x) != 0)
        >>> q.is_forall()
        False
        """
    return Z3_is_quantifier_forall(self.ctx_ref(), self.ast)