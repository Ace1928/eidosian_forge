from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_convolution_subset():
    assert convolution_subset([], []) == []
    assert convolution_subset([], [Rational(1, 3)]) == []
    assert convolution_subset([6 + I * 3 / 7], [Rational(2, 3)]) == [4 + I * 2 / 7]
    a = [1, Rational(5, 3), sqrt(3), 4 + 5 * I]
    b = [64, 71, 55, 47, 33, 29, 15]
    c = [3 + I * 2 / 3, 5 + 7 * I, 7, Rational(7, 5), 9]
    assert convolution_subset(a, b) == [64, Rational(533, 3), 55 + 64 * sqrt(3), 71 * sqrt(3) + Rational(1184, 3) + 320 * I, 33, 84, 15 + 33 * sqrt(3), 29 * sqrt(3) + 157 + 165 * I]
    assert convolution_subset(b, c) == [192 + I * 128 / 3, 533 + I * 1486 / 3, 613 + I * 110 / 3, Rational(5013, 5) + I * 1249 / 3, 675 + 22 * I, 891 + I * 751 / 3, 771 + 10 * I, Rational(3736, 5) + 105 * I]
    assert convolution_subset(a, c) == convolution_subset(c, a)
    assert convolution_subset(a[:2], b) == [64, Rational(533, 3), 55, Rational(416, 3), 33, 84, 15, 25]
    assert convolution_subset(a[:2], c) == [3 + I * 2 / 3, 10 + I * 73 / 9, 7, Rational(196, 15), 9, 15, 0, 0]
    u, v, w, x, y, z = symbols('u v w x y z')
    assert convolution_subset([u, v, w], [x, y]) == [u * x, u * y + v * x, w * x, w * y]
    assert convolution_subset([u, v, w, x], [y, z]) == [u * y, u * z + v * y, w * y, w * z + x * y]
    assert convolution_subset([u, v], [x, y, z]) == convolution_subset([x, y, z], [u, v])
    raises(TypeError, lambda: convolution_subset(x, z))
    raises(TypeError, lambda: convolution_subset(Rational(7, 3), u))