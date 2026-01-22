from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_cyclic_convolution():
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5)]
    b = [9, 5, 5, 4, 3, 2]
    assert convolution([1, 2, 3], [4, 5, 6], cycle=0) == convolution([1, 2, 3], [4, 5, 6], cycle=5) == convolution([1, 2, 3], [4, 5, 6])
    assert convolution([1, 2, 3], [4, 5, 6], cycle=3) == [31, 31, 28]
    a = [Rational(1, 3), Rational(7, 3), Rational(5, 9), Rational(2, 7), Rational(5, 8)]
    b = [Rational(3, 5), Rational(4, 7), Rational(7, 8), Rational(8, 9)]
    assert convolution(a, b, cycle=0) == convolution(a, b, cycle=len(a) + len(b) - 1)
    assert convolution(a, b, cycle=4) == [Rational(87277, 26460), Rational(30521, 11340), Rational(11125, 4032), Rational(3653, 1080)]
    assert convolution(a, b, cycle=6) == [Rational(20177, 20160), Rational(676, 315), Rational(47, 24), Rational(3053, 1080), Rational(16397, 5292), Rational(2497, 2268)]
    assert convolution(a, b, cycle=9) == convolution(a, b, cycle=0) + [S.Zero]
    a = [2313, 5323532, S(3232), 42142, 42242421]
    b = [S(33456), 56757, 45754, 432423]
    assert convolution(a, b, prime=19 * 2 ** 10 + 1, cycle=0) == convolution(a, b, prime=19 * 2 ** 10 + 1, cycle=8) == convolution(a, b, prime=19 * 2 ** 10 + 1)
    assert convolution(a, b, prime=19 * 2 ** 10 + 1, cycle=5) == [96, 17146, 2664, 15534, 3517]
    assert convolution(a, b, prime=19 * 2 ** 10 + 1, cycle=7) == [4643, 3458, 1260, 15534, 3517, 16314, 13688]
    assert convolution(a, b, prime=19 * 2 ** 10 + 1, cycle=9) == convolution(a, b, prime=19 * 2 ** 10 + 1) + [0]
    u, v, w, x, y = symbols('u v w x y')
    p, q, r, s, t = symbols('p q r s t')
    c = [u, v, w, x, y]
    d = [p, q, r, s, t]
    assert convolution(a, b, dyadic=True, cycle=3) == [2499522285783, 19861417974796, 4702176579021]
    assert convolution(a, b, dyadic=True, cycle=5) == [2718149225143, 2114320852171, 20571217906407, 246166418903, 1413262436976]
    assert convolution(c, d, dyadic=True, cycle=4) == [p * u + p * y + q * v + r * w + s * x + t * u + t * y, p * v + q * u + q * y + r * x + s * w + t * v, p * w + q * x + r * u + r * y + s * v + t * w, p * x + q * w + r * v + s * u + s * y + t * x]
    assert convolution(c, d, dyadic=True, cycle=6) == [p * u + q * v + r * w + r * y + s * x + t * w + t * y, p * v + q * u + r * x + s * w + s * y + t * x, p * w + q * x + r * u + s * v, p * x + q * w + r * v + s * u, p * y + t * u, q * y + t * v]
    assert convolution(a, b, subset=True, cycle=7) == [18266671799811, 178235365533, 213958794, 246166418903, 1413262436976, 2397553088697, 1932759730434]
    assert convolution(a[1:], b, subset=True, cycle=4) == [178104086592, 302255835516, 244982785880, 3717819845434]
    assert convolution(a, b[:-1], subset=True, cycle=6) == [1932837114162, 178235365533, 213958794, 245166224504, 1413262436976, 2397553088697]
    assert convolution(c, d, subset=True, cycle=3) == [p * u + p * x + q * w + r * v + r * y + s * u + t * w, p * v + p * y + q * u + s * y + t * u + t * x, p * w + q * y + r * u + t * v]
    assert convolution(c, d, subset=True, cycle=5) == [p * u + q * y + t * v, p * v + q * u + r * y + t * w, p * w + r * u + s * y + t * x, p * x + q * w + r * v + s * u, p * y + t * u]
    raises(ValueError, lambda: convolution([1, 2, 3], [4, 5, 6], cycle=-1))