from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_monomial_mul():
    assert sdm_monomial_mul((1, 1, 0), (1, 3)) == (1, 2, 3)