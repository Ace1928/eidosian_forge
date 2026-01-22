from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_ModuleElement_column():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    e = A(0)
    col1 = e.column()
    assert col1 == e.col and col1 is not e.col
    col2 = e.column(domain=FF(5))
    assert col2.domain.is_FF