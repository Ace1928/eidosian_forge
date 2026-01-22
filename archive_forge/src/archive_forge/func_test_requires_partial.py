from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.functions.special.bessel import besselj
from sympy.functions.special.polynomials import legendre
from sympy.functions.combinatorial.numbers import bell
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.testing.pytest import XFAIL
def test_requires_partial():
    x, y, z, t, nu = symbols('x y z t nu')
    n = symbols('n', integer=True)
    f = x * y
    assert requires_partial(Derivative(f, x)) is True
    assert requires_partial(Derivative(f, y)) is True
    assert requires_partial(Derivative(Integral(exp(-x * y), (x, 0, oo)), y, evaluate=False)) is False
    f = besselj(nu, x)
    assert requires_partial(Derivative(f, x)) is True
    assert requires_partial(Derivative(f, nu)) is True
    f = besselj(n, x)
    assert requires_partial(Derivative(f, x)) is False
    assert requires_partial(Derivative(f, n)) is False
    f = bell(n, x)
    assert requires_partial(Derivative(f, x)) is False
    assert requires_partial(Derivative(f, n)) is False
    f = legendre(0, x)
    assert requires_partial(Derivative(f, x)) is False
    f = legendre(n, x)
    assert requires_partial(Derivative(f, x)) is False
    assert requires_partial(Derivative(f, n)) is False
    f = x ** n
    assert requires_partial(Derivative(f, x)) is False
    assert requires_partial(Derivative(Integral((x * y) ** n * exp(-x * y), (x, 0, oo)), y, evaluate=False)) is False
    f = (exp(t), cos(t))
    g = sum(f)
    assert requires_partial(Derivative(g, t)) is False
    f = symbols('f', cls=Function)
    assert requires_partial(Derivative(f(x), x)) is False
    assert requires_partial(Derivative(f(x), y)) is False
    assert requires_partial(Derivative(f(x, y), x)) is True
    assert requires_partial(Derivative(f(x, y), y)) is True
    assert requires_partial(Derivative(f(x, y), z)) is True
    assert requires_partial(Derivative(f(x, y), x, y)) is True