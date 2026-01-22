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
def test_acot():
    assert acot(nan) is nan
    assert acot.nargs == FiniteSet(1)
    assert acot(-oo) == 0
    assert acot(oo) == 0
    assert acot(zoo) == 0
    assert acot(1) == pi / 4
    assert acot(0) == pi / 2
    assert acot(sqrt(3) / 3) == pi / 3
    assert acot(1 / sqrt(3)) == pi / 3
    assert acot(-1 / sqrt(3)) == -pi / 3
    assert acot(x).diff(x) == -1 / (1 + x ** 2)
    assert acot(r).is_extended_real is True
    assert acot(I * pi) == -I * acoth(pi)
    assert acot(-2 * I) == I * acoth(2)
    assert acot(x).is_positive is None
    assert acot(n).is_positive is False
    assert acot(p).is_positive is True
    assert acot(I).is_positive is False
    assert acot(Rational(1, 4)).is_rational is False
    assert unchanged(acot, cot(x))
    assert unchanged(acot, tan(x))
    assert acot(cot(Rational(1, 4))) == Rational(1, 4)
    assert acot(tan(Rational(-1, 4))) == Rational(1, 4) - pi / 2