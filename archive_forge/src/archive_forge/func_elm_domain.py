from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
from .singularities import singularities
from sympy.core import Pow, S
from sympy.core.function import diff, expand_mul
from sympy.core.kind import NumberKind
from sympy.core.mod import Mod
from sympy.core.numbers import equal_valued
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.polys.polytools import degree, lcm_list
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
from sympy.sets.fancysets import ImageSet
from sympy.utilities import filldedent
from sympy.utilities.iterables import iterable
def elm_domain(expr, intrvl):
    """ Finds the domain of an expression in any given interval """
    from sympy.solvers.solveset import solveset
    _start = intrvl.start
    _end = intrvl.end
    _singularities = solveset(expr.as_numer_denom()[1], symb, domain=S.Reals)
    if intrvl.right_open:
        if _end is S.Infinity:
            _domain1 = S.Reals
        else:
            _domain1 = solveset(expr < _end, symb, domain=S.Reals)
    else:
        _domain1 = solveset(expr <= _end, symb, domain=S.Reals)
    if intrvl.left_open:
        if _start is S.NegativeInfinity:
            _domain2 = S.Reals
        else:
            _domain2 = solveset(expr > _start, symb, domain=S.Reals)
    else:
        _domain2 = solveset(expr >= _start, symb, domain=S.Reals)
    expr_with_sing = Intersection(_domain1, _domain2)
    expr_domain = Complement(expr_with_sing, _singularities)
    return expr_domain