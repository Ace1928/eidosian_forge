from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises
def test_sin_expansion():
    assert sin(x + y).expand(trig=True) == sin(x) * cos(y) + cos(x) * sin(y)
    assert sin(x - y).expand(trig=True) == sin(x) * cos(y) - cos(x) * sin(y)
    assert sin(y - x).expand(trig=True) == cos(x) * sin(y) - sin(x) * cos(y)
    assert sin(2 * x).expand(trig=True) == 2 * sin(x) * cos(x)
    assert sin(3 * x).expand(trig=True) == -4 * sin(x) ** 3 + 3 * sin(x)
    assert sin(4 * x).expand(trig=True) == -8 * sin(x) ** 3 * cos(x) + 4 * sin(x) * cos(x)
    _test_extrig(sin, 2, 2 * sin(1) * cos(1))
    _test_extrig(sin, 3, -4 * sin(1) ** 3 + 3 * sin(1))