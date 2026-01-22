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
def test_csc():
    x = symbols('x', real=True)
    z = symbols('z')
    cosecant = csc('x')
    alternate = 1 / sin('x')
    assert cosecant.equals(alternate) == True
    assert alternate.equals(cosecant) == True
    assert csc.nargs == FiniteSet(1)
    assert csc(0) is zoo
    assert csc(pi) is zoo
    assert csc(zoo) is nan
    assert csc(pi / 2) == 1
    assert csc(-pi / 2) == -1
    assert csc(pi / 6) == 2
    assert csc(pi / 3) == 2 * sqrt(3) / 3
    assert csc(pi * Rational(5, 2)) == 1
    assert csc(pi * Rational(9, 7)) == -csc(pi * Rational(2, 7))
    assert csc(pi * Rational(3, 4)) == sqrt(2)
    assert csc(I) == -I / sinh(1)
    assert csc(x * I) == -I / sinh(x)
    assert csc(-x) == -csc(x)
    assert csc(acsc(x)) == x
    assert csc(z).conjugate() == csc(conjugate(z))
    assert csc(z).as_real_imag() == (sin(re(z)) * cosh(im(z)) / (sin(re(z)) ** 2 * cosh(im(z)) ** 2 + cos(re(z)) ** 2 * sinh(im(z)) ** 2), -cos(re(z)) * sinh(im(z)) / (sin(re(z)) ** 2 * cosh(im(z)) ** 2 + cos(re(z)) ** 2 * sinh(im(z)) ** 2))
    assert csc(x).expand(trig=True) == 1 / sin(x)
    assert csc(2 * x).expand(trig=True) == 1 / (2 * sin(x) * cos(x))
    assert csc(x).is_extended_real == True
    assert csc(z).is_real == None
    assert csc(a).is_algebraic is None
    assert csc(na).is_algebraic is False
    assert csc(x).as_leading_term() == csc(x)
    assert csc(0, evaluate=False).is_finite == False
    assert csc(x).is_finite == None
    assert csc(pi / 2, evaluate=False).is_finite == True
    assert series(csc(x), x, x0=pi / 2, n=6) == 1 + (x - pi / 2) ** 2 / 2 + 5 * (x - pi / 2) ** 4 / 24 + O((x - pi / 2) ** 6, (x, pi / 2))
    assert series(csc(x), x, x0=0, n=6) == 1 / x + x / 6 + 7 * x ** 3 / 360 + 31 * x ** 5 / 15120 + O(x ** 6)
    assert csc(x).diff(x) == -cot(x) * csc(x)
    assert csc(x).taylor_term(2, x) == 0
    assert csc(x).taylor_term(3, x) == 7 * x ** 3 / 360
    assert csc(x).taylor_term(5, x) == 31 * x ** 5 / 15120
    raises(ArgumentIndexError, lambda: csc(x).fdiff(2))