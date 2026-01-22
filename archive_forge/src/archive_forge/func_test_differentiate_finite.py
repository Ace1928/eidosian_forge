from itertools import product
from sympy.core.function import (Function, diff)
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.calculus.finite_diff import (
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_differentiate_finite():
    x, y, h = symbols('x y h')
    f = Function('f')
    with warns_deprecated_sympy():
        res0 = differentiate_finite(f(x, y) + exp(42), x, y, evaluate=True)
    xm, xp, ym, yp = [v + sign * S.Half for v, sign in product([x, y], [-1, 1])]
    ref0 = f(xm, ym) + f(xp, yp) - f(xm, yp) - f(xp, ym)
    assert (res0 - ref0).simplify() == 0
    g = Function('g')
    with warns_deprecated_sympy():
        res1 = differentiate_finite(f(x) * g(x) + 42, x, evaluate=True)
    ref1 = (-f(x - S.Half) + f(x + S.Half)) * g(x) + (-g(x - S.Half) + g(x + S.Half)) * f(x)
    assert (res1 - ref1).simplify() == 0
    res2 = differentiate_finite(f(x) + x ** 3 + 42, x, points=[x - 1, x + 1])
    ref2 = (f(x + 1) + (x + 1) ** 3 - f(x - 1) - (x - 1) ** 3) / 2
    assert (res2 - ref2).simplify() == 0
    raises(TypeError, lambda: differentiate_finite(f(x) * g(x), x, pints=[x - 1, x + 1]))
    res3 = differentiate_finite(f(x) * g(x).diff(x), x)
    ref3 = (-g(x) + g(x + 1)) * f(x + S.Half) - (g(x) - g(x - 1)) * f(x - S.Half)
    assert res3 == ref3
    res4 = differentiate_finite(f(x) * g(x).diff(x).diff(x), x)
    ref4 = -((g(x - Rational(3, 2)) - 2 * g(x - S.Half) + g(x + S.Half)) * f(x - S.Half)) + (g(x - S.Half) - 2 * g(x + S.Half) + g(x + Rational(3, 2))) * f(x + S.Half)
    assert res4 == ref4
    res5_expr = f(x).diff(x) * g(x).diff(x)
    res5 = differentiate_finite(res5_expr, points=[x - h, x, x + h])
    ref5 = (-2 * f(x) / h + f(-h + x) / (2 * h) + 3 * f(h + x) / (2 * h)) * (-2 * g(x) / h + g(-h + x) / (2 * h) + 3 * g(h + x) / (2 * h)) / (2 * h) - (2 * f(x) / h - 3 * f(-h + x) / (2 * h) - f(h + x) / (2 * h)) * (2 * g(x) / h - 3 * g(-h + x) / (2 * h) - g(h + x) / (2 * h)) / (2 * h)
    assert res5 == ref5
    res6 = res5.limit(h, 0).doit()
    ref6 = diff(res5_expr, x)
    assert res6 == ref6