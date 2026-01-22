import itertools as it
from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings
def test_Max():
    from sympy.abc import x, y, z
    n = Symbol('n', negative=True)
    n_ = Symbol('n_', negative=True)
    nn = Symbol('nn', nonnegative=True)
    p = Symbol('p', positive=True)
    p_ = Symbol('p_', positive=True)
    r = Symbol('r', real=True)
    assert Max(5, 4) == 5
    assert Max() is S.NegativeInfinity
    assert Max(x) == x
    assert Max(x, y) == Max(y, x)
    assert Max(x, y, z) == Max(z, y, x)
    assert Max(x, Max(y, z)) == Max(z, y, x)
    assert Max(x, Min(y, oo)) == Max(x, y)
    assert Max(n, -oo, n_, p, 2) == Max(p, 2)
    assert Max(n, -oo, n_, p) == p
    assert Max(2, x, p, n, -oo, S.NegativeInfinity, n_, p, 2) == Max(2, x, p)
    assert Max(0, x, 1, y) == Max(1, x, y)
    assert Max(r, r + 1, r - 1) == 1 + r
    assert Max(1000, 100, -100, x, p, n) == Max(p, x, 1000)
    assert Max(cos(x), sin(x)) == Max(sin(x), cos(x))
    assert Max(cos(x), sin(x)).subs(x, 1) == sin(1)
    assert Max(cos(x), sin(x)).subs(x, S.Half) == cos(S.Half)
    raises(ValueError, lambda: Max(cos(x), sin(x)).subs(x, I))
    raises(ValueError, lambda: Max(I))
    raises(ValueError, lambda: Max(I, x))
    raises(ValueError, lambda: Max(S.ComplexInfinity, 1))
    assert Max(n, -oo, n_, p, 2) == Max(p, 2)
    assert Max(n, -oo, n_, p, 1000) == Max(p, 1000)
    assert Max(1, x).diff(x) == Heaviside(x - 1)
    assert Max(x, 1).diff(x) == Heaviside(x - 1)
    assert Max(x ** 2, 1 + x, 1).diff(x) == 2 * x * Heaviside(x ** 2 - Max(1, x + 1)) + Heaviside(x - Max(1, x ** 2) + 1)
    e = Max(0, x)
    assert e.n().args == (0, x)
    m = Max(p, p_, n, r)
    assert m.is_positive is True
    assert m.is_nonnegative is True
    assert m.is_negative is False
    m = Max(n, n_)
    assert m.is_positive is False
    assert m.is_nonnegative is False
    assert m.is_negative is True
    m = Max(n, n_, r)
    assert m.is_positive is None
    assert m.is_nonnegative is None
    assert m.is_negative is None
    m = Max(n, nn, r)
    assert m.is_positive is None
    assert m.is_nonnegative is True
    assert m.is_negative is False