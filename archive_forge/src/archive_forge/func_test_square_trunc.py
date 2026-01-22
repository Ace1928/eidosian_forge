from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_square_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (1 + t * x + t * y) * 2
    p1 = rs_mul(p, p, x, 3)
    p2 = rs_square(p, x, 3)
    assert p1 == p2
    p = 1 + x + x ** 2 + x ** 3
    assert rs_square(p, x, 4) == 4 * x ** 3 + 3 * x ** 2 + 2 * x + 1