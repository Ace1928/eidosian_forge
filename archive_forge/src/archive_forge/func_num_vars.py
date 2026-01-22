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
def num_vars(self):
    """Return the number of variables bounded by this quantifier.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> x = Int('x')
        >>> y = Int('y')
        >>> q = ForAll([x, y], f(x, y) >= x)
        >>> q.num_vars()
        2
        """
    return int(Z3_get_quantifier_num_bound(self.ctx_ref(), self.ast))