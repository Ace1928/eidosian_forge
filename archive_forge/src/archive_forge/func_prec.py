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
def prec(self):
    """Return the precision (under-approximation, over-approximation, or precise) of the goal `self`.

        >>> g = Goal()
        >>> g.prec() == Z3_GOAL_PRECISE
        True
        >>> x, y = Ints('x y')
        >>> g.add(x == y + 1)
        >>> g.prec() == Z3_GOAL_PRECISE
        True
        >>> t  = With(Tactic('add-bounds'), add_bound_lower=0, add_bound_upper=10)
        >>> g2 = t(g)[0]
        >>> g2
        [x == y + 1, x <= 10, x >= 0, y <= 10, y >= 0]
        >>> g2.prec() == Z3_GOAL_PRECISE
        False
        >>> g2.prec() == Z3_GOAL_UNDER
        True
        """
    return Z3_goal_precision(self.ctx.ref(), self.goal)