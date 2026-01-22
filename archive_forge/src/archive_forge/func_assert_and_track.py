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
def assert_and_track(self, a, p):
    """Assert constraint `a` and track it in the unsat core using the Boolean constant `p`.

        If `p` is a string, it will be automatically converted into a Boolean constant.

        >>> x = Int('x')
        >>> p3 = Bool('p3')
        >>> s = Optimize()
        >>> s.assert_and_track(x > 0,  'p1')
        >>> s.assert_and_track(x != 1, 'p2')
        >>> s.assert_and_track(x < 0,  p3)
        >>> print(s.check())
        unsat
        >>> c = s.unsat_core()
        >>> len(c)
        2
        >>> Bool('p1') in c
        True
        >>> Bool('p2') in c
        False
        >>> p3 in c
        True
        """
    if isinstance(p, str):
        p = Bool(p, self.ctx)
    _z3_assert(isinstance(a, BoolRef), 'Boolean expression expected')
    _z3_assert(isinstance(p, BoolRef) and is_const(p), 'Boolean expression expected')
    Z3_optimize_assert_and_track(self.ctx.ref(), self.optimize, a.as_ast(), p.as_ast())