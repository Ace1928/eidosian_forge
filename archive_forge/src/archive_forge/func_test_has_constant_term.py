from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_has_constant_term():
    R, x, y, z = ring('x, y, z', QQ)
    p = y + x * z
    assert _has_constant_term(p, x)
    p = x + x ** 4
    assert not _has_constant_term(p, x)
    p = 1 + x + x ** 4
    assert _has_constant_term(p, x)
    p = x + y + x * z