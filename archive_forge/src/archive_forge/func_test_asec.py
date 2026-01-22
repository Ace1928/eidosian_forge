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
def test_asec():
    z = Symbol('z', zero=True)
    assert asec(z) is zoo
    assert asec(nan) is nan
    assert asec(1) == 0
    assert asec(-1) == pi
    assert asec(oo) == pi / 2
    assert asec(-oo) == pi / 2
    assert asec(zoo) == pi / 2
    assert asec(sec(pi * Rational(13, 4))) == pi * Rational(3, 4)
    assert asec(1 + sqrt(5)) == pi * Rational(2, 5)
    assert asec(2 / sqrt(3)) == pi / 6
    assert asec(sqrt(4 - 2 * sqrt(2))) == pi / 8
    assert asec(-sqrt(4 + 2 * sqrt(2))) == pi * Rational(5, 8)
    assert asec(sqrt(2 + 2 * sqrt(5) / 5)) == pi * Rational(3, 10)
    assert asec(-sqrt(2 + 2 * sqrt(5) / 5)) == pi * Rational(7, 10)
    assert asec(sqrt(2) - sqrt(6)) == pi * Rational(11, 12)
    assert asec(x).diff(x) == 1 / (x ** 2 * sqrt(1 - 1 / x ** 2))
    assert asec(x).rewrite(log) == I * log(sqrt(1 - 1 / x ** 2) + I / x) + pi / 2
    assert asec(x).rewrite(asin) == -asin(1 / x) + pi / 2
    assert asec(x).rewrite(acos) == acos(1 / x)
    assert asec(x).rewrite(atan) == pi * (1 - sqrt(x ** 2) / x) / 2 + sqrt(x ** 2) * atan(sqrt(x ** 2 - 1)) / x
    assert asec(x).rewrite(acot) == pi * (1 - sqrt(x ** 2) / x) / 2 + sqrt(x ** 2) * acot(1 / sqrt(x ** 2 - 1)) / x
    assert asec(x).rewrite(acsc) == -acsc(x) + pi / 2
    raises(ArgumentIndexError, lambda: asec(x).fdiff(2))