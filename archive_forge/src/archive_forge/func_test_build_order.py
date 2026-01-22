from sympy.polys.domains import QQ, ZZ
from sympy.polys.polyerrors import ExactQuotientFailed, CoercionFailed, NotReversible
from sympy.abc import x, y
from sympy.testing.pytest import raises
def test_build_order():
    R = QQ.old_poly_ring(x, y, order=(('lex', x), ('ilex', y)))
    assert R.order((1, 5)) == ((1,), (-5,))