from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_basis_elements():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    basis = B.basis_elements()
    bp = B.basis_element_pullbacks()
    for i, (e, p) in enumerate(zip(basis, bp)):
        c = [0] * 4
        assert e.module == B
        assert p.module == A
        c[i] = 1
        assert e == B(to_col(c))
        c[i] = 2
        assert p == A(to_col(c))