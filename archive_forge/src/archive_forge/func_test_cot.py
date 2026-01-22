from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_cot():
    R, x, y = ring('x, y', QQ)
    assert rs_cot(x ** 6 + x ** 7, x, 8) == x ** (-6) - x ** (-5) + x ** (-4) - x ** (-3) + x ** (-2) - x ** (-1) + 1 - x + x ** 2 - x ** 3 + x ** 4 - x ** 5 + 2 * x ** 6 / 3 - 4 * x ** 7 / 3
    assert rs_cot(x + x ** 2 * y, x, 5) == -x ** 4 * y ** 5 - x ** 4 * y / 15 + x ** 3 * y ** 4 - x ** 3 / 45 - x ** 2 * y ** 3 - x ** 2 * y / 3 + x * y ** 2 - x / 3 - y + x ** (-1)