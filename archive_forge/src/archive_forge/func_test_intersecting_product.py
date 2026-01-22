from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_intersecting_product():
    assert intersecting_product([], []) == []
    assert intersecting_product([], [Rational(1, 3)]) == []
    assert intersecting_product([6 + I * 3 / 7], [Rational(2, 3)]) == [4 + I * 2 / 7]
    a = [1, sqrt(5), Rational(3, 8) + 5 * I, 4 + 7 * I]
    b = [67, 51, 65, 48, 36, 79, 27]
    c = [3 + I * 2 / 5, 5 + 9 * I, 7, Rational(7, 19), 13]
    assert intersecting_product(a, b) == [195 * sqrt(5) + Rational(6979, 8) + 1886 * I, 178 * sqrt(5) + 520 + 910 * I, Rational(841, 2) + 1344 * I, 192 + 336 * I, 0, 0, 0, 0]
    assert intersecting_product(b, c) == [Rational(128553, 19) + I * 9521 / 5, Rational(17820, 19) + 1602 * I, Rational(19264, 19), Rational(336, 19), 1846, 0, 0, 0]
    assert intersecting_product(a, c) == intersecting_product(c, a)
    assert intersecting_product(b[1:], c[:-1]) == [Rational(64788, 19) + I * 8622 / 5, Rational(12804, 19) + 1152 * I, Rational(11508, 19), Rational(252, 19), 0, 0, 0, 0]
    assert intersecting_product(a, c[:-2]) == [Rational(-99, 5) + 10 * sqrt(5) + 2 * sqrt(5) * I / 5 + I * 3021 / 40, -43 + 5 * sqrt(5) + 9 * sqrt(5) * I + 71 * I, Rational(245, 8) + 84 * I, 0]
    u, v, w, x, y, z = symbols('u v w x y z')
    assert intersecting_product([u, v, w], [x, y]) == [u * x + u * y + v * x + w * x + w * y, v * y, 0, 0]
    assert intersecting_product([u, v, w, x], [y, z]) == [u * y + u * z + v * y + w * y + w * z + x * y, v * z + x * z, 0, 0]
    assert intersecting_product([u, v], [x, y, z]) == intersecting_product([x, y, z], [u, v])
    raises(TypeError, lambda: intersecting_product(x, z))
    raises(TypeError, lambda: intersecting_product(u, Rational(8, 3)))