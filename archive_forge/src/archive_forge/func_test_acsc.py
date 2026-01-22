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
def test_acsc():
    assert acsc(nan) is nan
    assert acsc(1) == pi / 2
    assert acsc(-1) == -pi / 2
    assert acsc(oo) == 0
    assert acsc(-oo) == 0
    assert acsc(zoo) == 0
    assert acsc(0) is zoo
    assert acsc(csc(3)) == -3 + pi
    assert acsc(csc(4)) == -4 + pi
    assert acsc(csc(6)) == 6 - 2 * pi
    assert unchanged(acsc, csc(x))
    assert unchanged(acsc, sec(x))
    assert acsc(2 / sqrt(3)) == pi / 3
    assert acsc(csc(pi * Rational(13, 4))) == -pi / 4
    assert acsc(sqrt(2 + 2 * sqrt(5) / 5)) == pi / 5
    assert acsc(-sqrt(2 + 2 * sqrt(5) / 5)) == -pi / 5
    assert acsc(-2) == -pi / 6
    assert acsc(-sqrt(4 + 2 * sqrt(2))) == -pi / 8
    assert acsc(sqrt(4 - 2 * sqrt(2))) == pi * Rational(3, 8)
    assert acsc(1 + sqrt(5)) == pi / 10
    assert acsc(sqrt(2) - sqrt(6)) == pi * Rational(-5, 12)
    assert acsc(x).diff(x) == -1 / (x ** 2 * sqrt(1 - 1 / x ** 2))
    assert acsc(x).rewrite(log) == -I * log(sqrt(1 - 1 / x ** 2) + I / x)
    assert acsc(x).rewrite(asin) == asin(1 / x)
    assert acsc(x).rewrite(acos) == -acos(1 / x) + pi / 2
    assert acsc(x).rewrite(atan) == (-atan(sqrt(x ** 2 - 1)) + pi / 2) * sqrt(x ** 2) / x
    assert acsc(x).rewrite(acot) == (-acot(1 / sqrt(x ** 2 - 1)) + pi / 2) * sqrt(x ** 2) / x
    assert acsc(x).rewrite(asec) == -asec(x) + pi / 2
    raises(ArgumentIndexError, lambda: acsc(x).fdiff(2))