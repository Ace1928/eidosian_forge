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
@slow
def test_sincos_rewrite_sqrt():
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t * p
            for i in range(1, min((n + 1) // 2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i * pi / n
                    s1 = sin(x).rewrite(sqrt)
                    c1 = cos(x).rewrite(sqrt)
                    assert not s1.has(cos, sin), 'fails for %d*pi/%d' % (i, n)
                    assert not c1.has(cos, sin), 'fails for %d*pi/%d' % (i, n)
                    assert 0.001 > abs(sin(x.evalf(5)) - s1.evalf(2)), 'fails for %d*pi/%d' % (i, n)
                    assert 0.001 > abs(cos(x.evalf(5)) - c1.evalf(2)), 'fails for %d*pi/%d' % (i, n)
    assert cos(pi / 14).rewrite(sqrt) == sqrt(cos(pi / 7) / 2 + S.Half)
    assert cos(pi * Rational(-15, 2) / 11, evaluate=False).rewrite(sqrt) == -sqrt(-cos(pi * Rational(4, 11)) / 2 + S.Half)
    assert cos(Mul(2, pi, S.Half, evaluate=False), evaluate=False).rewrite(sqrt) == -1
    e = cos(pi / 3 / 17)
    a = -3 * sqrt(-sqrt(17) + 17) * sqrt(sqrt(17) + 17) / 64 - 3 * sqrt(34) * sqrt(sqrt(17) + 17) / 128 - sqrt(sqrt(17) + 17) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 64 - sqrt(-sqrt(17) + 17) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 128 - Rational(1, 32) + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 64 + 3 * sqrt(2) * sqrt(sqrt(17) + 17) / 128 + sqrt(34) * sqrt(-sqrt(17) + 17) / 128 + 13 * sqrt(2) * sqrt(-sqrt(17) + 17) / 128 + sqrt(17) * sqrt(-sqrt(17) + 17) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 128 + 5 * sqrt(17) / 32 + sqrt(3) * sqrt(-sqrt(2) * sqrt(sqrt(17) + 17) * sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) / 8 - 5 * sqrt(2) * sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 64 - 3 * sqrt(2) * sqrt(-sqrt(17) + 17) * sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) / 32 + sqrt(34) * sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 64 + sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) / 2 + S.Half + sqrt(-sqrt(17) + 17) * sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + sqrt(34) * sqrt(-sqrt(17) + 17) * sqrt(sqrt(17) / 32 + sqrt(2) * sqrt(-sqrt(17) + 17) / 32 + sqrt(2) * sqrt(-8 * sqrt(2) * sqrt(sqrt(17) + 17) - sqrt(2) * sqrt(-sqrt(17) + 17) + sqrt(34) * sqrt(-sqrt(17) + 17) + 6 * sqrt(17) + 34) / 32 + Rational(15, 32)) / 32) / 2
    assert e.rewrite(sqrt) == a
    assert e.n() == a.n()
    assert cos(pi / 9 / 17).rewrite(sqrt) == sin(pi / 9) * sin(pi * Rational(2, 17)) + cos(pi / 9) * cos(pi * Rational(2, 17))