from sympy.polys.monomials import (
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.abc import a, b, c, x, y, z
from sympy.core import S, symbols
from sympy.testing.pytest import raises
def test_monomial_max():
    assert monomial_max((3, 4, 5), (0, 5, 1), (6, 3, 9)) == (6, 5, 9)