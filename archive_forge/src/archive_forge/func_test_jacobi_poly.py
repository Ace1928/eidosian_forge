from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.orthopolys import (
from sympy.abc import x, a, b
def test_jacobi_poly():
    raises(ValueError, lambda: jacobi_poly(-1, a, b, x))
    assert jacobi_poly(1, a, b, x, polys=True) == Poly((a / 2 + b / 2 + 1) * x + a / 2 - b / 2, x, domain='ZZ(a,b)')
    assert jacobi_poly(0, a, b, x) == 1
    assert jacobi_poly(1, a, b, x) == a / 2 - b / 2 + x * (a / 2 + b / 2 + 1)
    assert jacobi_poly(2, a, b, x) == a ** 2 / 8 - a * b / 4 - a / 8 + b ** 2 / 8 - b / 8 + x ** 2 * (a ** 2 / 8 + a * b / 4 + a * Q(7, 8) + b ** 2 / 8 + b * Q(7, 8) + Q(3, 2)) + x * (a ** 2 / 4 + a * Q(3, 4) - b ** 2 / 4 - b * Q(3, 4)) - S.Half
    assert jacobi_poly(1, a, b, polys=True) == Poly((a / 2 + b / 2 + 1) * x + a / 2 - b / 2, x, domain='ZZ(a,b)')