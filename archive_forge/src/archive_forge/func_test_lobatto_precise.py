from sympy.core import S, Rational
from sympy.integrals.quadrature import (gauss_legendre, gauss_laguerre,
def test_lobatto_precise():
    x, w = gauss_lobatto(3, 40)
    assert [str(r) for r in x] == ['-1', '0', '1']
    assert [str(r) for r in w] == ['0.3333333333333333333333333333333333333333', '1.333333333333333333333333333333333333333', '0.3333333333333333333333333333333333333333']