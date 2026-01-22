from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_puiseux_algebraic():
    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.from_sympy(sqrt(2))
    x, y = symbols('x, y')
    R, xr, yr = ring([x, y], K)
    p = (1 + sqrt2) * xr ** QQ(1, 2) + (1 - sqrt2) * yr ** QQ(2, 3)
    assert dict(p) == {(QQ(1, 2), QQ(0)): 1 + sqrt2, (QQ(0), QQ(2, 3)): 1 - sqrt2}
    assert p.as_expr() == (1 + sqrt(2)) * x ** (S(1) / 2) + (1 - sqrt(2)) * y ** (S(2) / 3)