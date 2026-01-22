from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def build_ideal(x, terms):
    """
        Build generators for our ideal. ``Terms`` is an iterable with elements of
        the form (fn, coeff), indicating that we have a generator fn(coeff*x).

        If any of the terms is trigonometric, sin(x) and cos(x) are guaranteed
        to appear in terms. Similarly for hyperbolic functions. For tan(n*x),
        sin(n*x) and cos(n*x) are guaranteed.
        """
    I = []
    y = Dummy('y')
    for fn, coeff in terms:
        for c, s, t, rel in ([cos, sin, tan, cos(x) ** 2 + sin(x) ** 2 - 1], [cosh, sinh, tanh, cosh(x) ** 2 - sinh(x) ** 2 - 1]):
            if coeff == 1 and fn in [c, s]:
                I.append(rel)
            elif fn == t:
                I.append(t(coeff * x) * c(coeff * x) - s(coeff * x))
            elif fn in [c, s]:
                cn = fn(coeff * y).expand(trig=True).subs(y, x)
                I.append(fn(coeff * x) - cn)
    return list(set(I))