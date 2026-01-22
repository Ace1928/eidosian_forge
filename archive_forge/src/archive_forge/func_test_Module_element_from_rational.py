from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_element_from_rational():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    rA = A.element_from_rational(QQ(22, 7))
    rB = B.element_from_rational(QQ(22, 7))
    assert rA.coeffs == [22, 0, 0, 0]
    assert rA.denom == 7
    assert rA.module == A
    assert rB.coeffs == [22, 0, 0, 0]
    assert rB.denom == 7
    assert rB.module == A