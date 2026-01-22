from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_is_regular():
    R, x, y = ring('x, y', QQ)
    p = 1 + 2 * x + x ** 2 + 3 * x ** 3
    assert not rs_is_puiseux(p, x)
    p = x + x ** QQ(1, 5) * y
    assert rs_is_puiseux(p, x)
    assert not rs_is_puiseux(p, y)
    p = x + x ** 2 * y ** QQ(1, 5) * y
    assert not rs_is_puiseux(p, x)