from sympy.concrete.products import Product
from sympy.core.function import expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.factorials import subfactorial
from sympy.functions.special.gamma_functions import uppergamma
from sympy.testing.pytest import XFAIL, raises, slow
def test_rf_eval_apply():
    x, y = symbols('x,y')
    n, k = symbols('n k', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)
    assert rf(nan, y) is nan
    assert rf(x, nan) is nan
    assert unchanged(rf, x, y)
    assert rf(oo, 0) == 1
    assert rf(-oo, 0) == 1
    assert rf(oo, 6) is oo
    assert rf(-oo, 7) is -oo
    assert rf(-oo, 6) is oo
    assert rf(oo, -6) is oo
    assert rf(-oo, -7) is oo
    assert rf(-1, pi) == 0
    assert rf(-5, 1 + I) == 0
    assert unchanged(rf, -3, k)
    assert unchanged(rf, x, Symbol('k', integer=False))
    assert rf(-3, Symbol('k', integer=False)) == 0
    assert rf(Symbol('x', negative=True, integer=True), Symbol('k', integer=False)) == 0
    assert rf(x, 0) == 1
    assert rf(x, 1) == x
    assert rf(x, 2) == x * (x + 1)
    assert rf(x, 3) == x * (x + 1) * (x + 2)
    assert rf(x, 5) == x * (x + 1) * (x + 2) * (x + 3) * (x + 4)
    assert rf(x, -1) == 1 / (x - 1)
    assert rf(x, -2) == 1 / ((x - 1) * (x - 2))
    assert rf(x, -3) == 1 / ((x - 1) * (x - 2) * (x - 3))
    assert rf(1, 100) == factorial(100)
    assert rf(x ** 2 + 3 * x, 2) == (x ** 2 + 3 * x) * (x ** 2 + 3 * x + 1)
    assert isinstance(rf(x ** 2 + 3 * x, 2), Mul)
    assert rf(x ** 3 + x, -2) == 1 / ((x ** 3 + x - 1) * (x ** 3 + x - 2))
    assert rf(Poly(x ** 2 + 3 * x, x), 2) == Poly(x ** 4 + 8 * x ** 3 + 19 * x ** 2 + 12 * x, x)
    assert isinstance(rf(Poly(x ** 2 + 3 * x, x), 2), Poly)
    raises(ValueError, lambda: rf(Poly(x ** 2 + 3 * x, x, y), 2))
    assert rf(Poly(x ** 3 + x, x), -2) == 1 / (x ** 6 - 9 * x ** 5 + 35 * x ** 4 - 75 * x ** 3 + 94 * x ** 2 - 66 * x + 20)
    raises(ValueError, lambda: rf(Poly(x ** 3 + x, x, y), -2))
    assert rf(x, m).is_integer is None
    assert rf(n, k).is_integer is None
    assert rf(n, m).is_integer is True
    assert rf(n, k + pi).is_integer is False
    assert rf(n, m + pi).is_integer is False
    assert rf(pi, m).is_integer is False

    def check(x, k, o, n):
        a, b = (Dummy(), Dummy())
        r = lambda x, k: o(a, b).rewrite(n).subs({a: x, b: k})
        for i in range(-5, 5):
            for j in range(-5, 5):
                assert o(i, j) == r(i, j), (o, n, i, j)
    check(x, k, rf, ff)
    check(x, k, rf, binomial)
    check(n, k, rf, factorial)
    check(x, y, rf, factorial)
    check(x, y, rf, binomial)
    assert rf(x, k).rewrite(ff) == ff(x + k - 1, k)
    assert rf(x, k).rewrite(gamma) == Piecewise((gamma(k + x) / gamma(x), x > 0), ((-1) ** k * gamma(1 - x) / gamma(-k - x + 1), True))
    assert rf(5, k).rewrite(gamma) == gamma(k + 5) / 24
    assert rf(x, k).rewrite(binomial) == factorial(k) * binomial(x + k - 1, k)
    assert rf(n, k).rewrite(factorial) == Piecewise((factorial(k + n - 1) / factorial(n - 1), n > 0), ((-1) ** k * factorial(-n) / factorial(-k - n), True))
    assert rf(5, k).rewrite(factorial) == factorial(k + 4) / 24
    assert rf(x, y).rewrite(factorial) == rf(x, y)
    assert rf(x, y).rewrite(binomial) == rf(x, y)
    import random
    from mpmath import rf as mpmath_rf
    for i in range(100):
        x = -500 + 500 * random.random()
        k = -500 + 500 * random.random()
        assert abs(mpmath_rf(x, k) - rf(x, k)) < 10 ** (-15)