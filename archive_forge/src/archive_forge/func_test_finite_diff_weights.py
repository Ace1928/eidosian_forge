from itertools import product
from sympy.core.function import (Function, diff)
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.calculus.finite_diff import (
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_finite_diff_weights():
    d = finite_diff_weights(1, [5, 6, 7], 5)
    assert d[1][2] == [Rational(-3, 2), 2, Rational(-1, 2)]
    xl = [0, 1, -1, 2, -2, 3, -3, 4, -4]
    d = finite_diff_weights(4, xl, S.Zero)
    for i in range(5):
        assert d[0][i] == [S.One] + [S.Zero] * 8
    assert d[1][0] == [S.Zero] * 9
    assert d[1][2] == [S.Zero, S.Half, Rational(-1, 2)] + [S.Zero] * 6
    assert d[1][4] == [S.Zero, Rational(2, 3), Rational(-2, 3), Rational(-1, 12), Rational(1, 12)] + [S.Zero] * 4
    assert d[1][6] == [S.Zero, Rational(3, 4), Rational(-3, 4), Rational(-3, 20), Rational(3, 20), Rational(1, 60), Rational(-1, 60)] + [S.Zero] * 2
    assert d[1][8] == [S.Zero, Rational(4, 5), Rational(-4, 5), Rational(-1, 5), Rational(1, 5), Rational(4, 105), Rational(-4, 105), Rational(-1, 280), Rational(1, 280)]
    for i in range(2):
        assert d[2][i] == [S.Zero] * 9
    assert d[2][2] == [-S(2), S.One, S.One] + [S.Zero] * 6
    assert d[2][4] == [Rational(-5, 2), Rational(4, 3), Rational(4, 3), Rational(-1, 12), Rational(-1, 12)] + [S.Zero] * 4
    assert d[2][6] == [Rational(-49, 18), Rational(3, 2), Rational(3, 2), Rational(-3, 20), Rational(-3, 20), Rational(1, 90), Rational(1, 90)] + [S.Zero] * 2
    assert d[2][8] == [Rational(-205, 72), Rational(8, 5), Rational(8, 5), Rational(-1, 5), Rational(-1, 5), Rational(8, 315), Rational(8, 315), Rational(-1, 560), Rational(-1, 560)]
    for i in range(3):
        assert d[3][i] == [S.Zero] * 9
    assert d[3][4] == [S.Zero, -S.One, S.One, S.Half, Rational(-1, 2)] + [S.Zero] * 4
    assert d[3][6] == [S.Zero, Rational(-13, 8), Rational(13, 8), S.One, -S.One, Rational(-1, 8), Rational(1, 8)] + [S.Zero] * 2
    assert d[3][8] == [S.Zero, Rational(-61, 30), Rational(61, 30), Rational(169, 120), Rational(-169, 120), Rational(-3, 10), Rational(3, 10), Rational(7, 240), Rational(-7, 240)]
    for i in range(4):
        assert d[4][i] == [S.Zero] * 9
    assert d[4][4] == [S(6), -S(4), -S(4), S.One, S.One] + [S.Zero] * 4
    assert d[4][6] == [Rational(28, 3), Rational(-13, 2), Rational(-13, 2), S(2), S(2), Rational(-1, 6), Rational(-1, 6)] + [S.Zero] * 2
    assert d[4][8] == [Rational(91, 8), Rational(-122, 15), Rational(-122, 15), Rational(169, 60), Rational(169, 60), Rational(-2, 5), Rational(-2, 5), Rational(7, 240), Rational(7, 240)]
    xl = [[j / S(2) for j in list(range(-i * 2 + 1, 0, 2)) + list(range(1, i * 2 + 1, 2))] for i in range(1, 5)]
    d = [finite_diff_weights({0: 1, 1: 2, 2: 4, 3: 4}[i], xl[i], 0) for i in range(4)]
    assert d[0][0][1] == [S.Half, S.Half]
    assert d[1][0][3] == [Rational(-1, 16), Rational(9, 16), Rational(9, 16), Rational(-1, 16)]
    assert d[2][0][5] == [Rational(3, 256), Rational(-25, 256), Rational(75, 128), Rational(75, 128), Rational(-25, 256), Rational(3, 256)]
    assert d[3][0][7] == [Rational(-5, 2048), Rational(49, 2048), Rational(-245, 2048), Rational(1225, 2048), Rational(1225, 2048), Rational(-245, 2048), Rational(49, 2048), Rational(-5, 2048)]
    assert d[0][1][1] == [-S.One, S.One]
    assert d[1][1][3] == [Rational(1, 24), Rational(-9, 8), Rational(9, 8), Rational(-1, 24)]
    assert d[2][1][5] == [Rational(-3, 640), Rational(25, 384), Rational(-75, 64), Rational(75, 64), Rational(-25, 384), Rational(3, 640)]
    assert d[3][1][7] == [Rational(5, 7168), Rational(-49, 5120), Rational(245, 3072), Rational(-1225, 1024), Rational(1225, 1024), Rational(-245, 3072), Rational(49, 5120), Rational(-5, 7168)]
    raises(ValueError, lambda: finite_diff_weights(-1, [1, 2]))
    raises(ValueError, lambda: finite_diff_weights(1.2, [1, 2]))
    x = symbols('x')
    raises(ValueError, lambda: finite_diff_weights(x, [1, 2]))