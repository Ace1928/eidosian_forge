from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_convolution():
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5)]
    b = [9, 5, 5, 4, 3, 2]
    c = [3, 5, 3, 7, 8]
    d = [1422, 6572, 3213, 5552]
    assert convolution(a, b) == convolution_fft(a, b)
    assert convolution(a, b, dps=9) == convolution_fft(a, b, dps=9)
    assert convolution(a, d, dps=7) == convolution_fft(d, a, dps=7)
    assert convolution(a, d[1:], dps=3) == convolution_fft(d[1:], a, dps=3)
    p = 7 * 17 * 2 ** 23 + 1
    q = 19 * 2 ** 10 + 1
    assert convolution(d, b, prime=q) == convolution_ntt(b, d, prime=q)
    assert convolution(c, b, prime=p) == convolution_ntt(b, c, prime=p)
    assert convolution(d, c, prime=p) == convolution_ntt(c, d, prime=p)
    raises(TypeError, lambda: convolution(b, d, dps=5, prime=q))
    raises(TypeError, lambda: convolution(b, d, dps=6, prime=q))
    assert convolution(a, b, dyadic=True) == convolution_fwht(a, b)
    assert convolution(a, b, dyadic=False) == convolution(a, b)
    raises(TypeError, lambda: convolution(b, d, dps=2, dyadic=True))
    raises(TypeError, lambda: convolution(b, d, prime=p, dyadic=True))
    raises(TypeError, lambda: convolution(a, b, dps=2, dyadic=True))
    raises(TypeError, lambda: convolution(b, c, prime=p, dyadic=True))
    assert convolution(a, b, subset=True) == convolution_subset(a, b) == convolution(a, b, subset=True, dyadic=False) == convolution(a, b, subset=True)
    assert convolution(a, b, subset=False) == convolution(a, b)
    raises(TypeError, lambda: convolution(a, b, subset=True, dyadic=True))
    raises(TypeError, lambda: convolution(c, d, subset=True, dps=6))
    raises(TypeError, lambda: convolution(a, c, subset=True, prime=q))