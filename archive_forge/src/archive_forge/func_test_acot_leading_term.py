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
def test_acot_leading_term():
    assert acot(1 / x).as_leading_term(x) == x
    assert acot(x + I).as_leading_term(x, cdir=1) == I * log(x) / 2 + pi / 4 - I * log(2) / 2
    assert acot(x + I).as_leading_term(x, cdir=-1) == I * log(x) / 2 + pi / 4 - I * log(2) / 2
    assert acot(x - I).as_leading_term(x, cdir=1) == -I * log(x) / 2 + pi / 4 + I * log(2) / 2
    assert acot(x - I).as_leading_term(x, cdir=-1) == -I * log(x) / 2 - 3 * pi / 4 + I * log(2) / 2
    assert acot(x).as_leading_term(x, cdir=1) == pi / 2
    assert acot(x).as_leading_term(x, cdir=-1) == -pi / 2
    assert acot(x + I / 2).as_leading_term(x, cdir=1) == pi - I * acoth(S(1) / 2)
    assert acot(x + I / 2).as_leading_term(x, cdir=-1) == -I * acoth(S(1) / 2)
    assert acot(x - I / 2).as_leading_term(x, cdir=1) == I * acoth(S(1) / 2)
    assert acot(x - I / 2).as_leading_term(x, cdir=-1) == -pi + I * acoth(S(1) / 2)
    assert acot(I / 2 - I * x - x ** 2).as_leading_term(x, cdir=1) == -pi / 2 - I * log(3) / 2
    assert acot(I / 2 - I * x - x ** 2).as_leading_term(x, cdir=-1) == -pi / 2 - I * log(3) / 2