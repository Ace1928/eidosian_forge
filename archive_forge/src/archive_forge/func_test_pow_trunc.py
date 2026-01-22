from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_pow_trunc():
    R, x, y, z = ring('x, y, z', QQ)
    p0 = y + x * z
    p = p0 ** 16
    for xx in (x, y, z):
        p1 = rs_trunc(p, xx, 8)
        p2 = rs_pow(p0, 16, xx, 8)
        assert p1 == p2
    p = 1 + x
    p1 = rs_pow(p, 3, x, 2)
    assert p1 == 1 + 3 * x
    assert rs_pow(p, 0, x, 2) == 1
    assert rs_pow(p, -2, x, 2) == 1 - 2 * x
    p = x + y
    assert rs_pow(p, 3, y, 3) == x ** 3 + 3 * x ** 2 * y + 3 * x * y ** 2
    assert rs_pow(1 + x, Rational(2, 3), x, 4) == 4 * x ** 3 / 81 - x ** 2 / 9 + x * Rational(2, 3) + 1