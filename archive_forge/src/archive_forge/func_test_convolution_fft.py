from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_convolution_fft():
    assert all((convolution_fft([], x, dps=y) == [] for x in ([], [1]) for y in (None, 3)))
    assert convolution_fft([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
    assert convolution_fft([1], [5, 6, 7]) == [5, 6, 7]
    assert convolution_fft([1, 3], [5, 6, 7]) == [5, 21, 25, 21]
    assert convolution_fft([1 + 2 * I], [2 + 3 * I]) == [-4 + 7 * I]
    assert convolution_fft([1 + 2 * I, 3 + 4 * I, 5 + 3 * I / 5], [Rational(2, 5) + 4 * I / 7]) == [Rational(-26, 35) + I * 48 / 35, Rational(-38, 35) + I * 116 / 35, Rational(58, 35) + I * 542 / 175]
    assert convolution_fft([Rational(3, 4), Rational(5, 6)], [Rational(7, 8), Rational(1, 3), Rational(2, 5)]) == [Rational(21, 32), Rational(47, 48), Rational(26, 45), Rational(1, 3)]
    assert convolution_fft([Rational(1, 9), Rational(2, 3), Rational(3, 5)], [Rational(2, 5), Rational(3, 7), Rational(4, 9)]) == [Rational(2, 45), Rational(11, 35), Rational(8152, 14175), Rational(523, 945), Rational(4, 15)]
    assert convolution_fft([pi, E, sqrt(2)], [sqrt(3), 1 / pi, 1 / E]) == [sqrt(3) * pi, 1 + sqrt(3) * E, E / pi + pi * exp(-1) + sqrt(6), sqrt(2) / pi + 1, sqrt(2) * exp(-1)]
    assert convolution_fft([2321, 33123], [5321, 6321, 71323]) == [12350041, 190918524, 374911166, 2362431729]
    assert convolution_fft([312313, 31278232], [32139631, 319631]) == [10037624576503, 1005370659728895, 9997492572392]
    raises(TypeError, lambda: convolution_fft(x, y))
    raises(ValueError, lambda: convolution_fft([x, y], [y, x]))