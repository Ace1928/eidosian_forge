from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Submodule_discard_before():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    B.compute_mult_tab()
    C = B.discard_before(2)
    assert C.parent == B.parent
    assert B.is_sq_maxrank_HNF() and (not C.is_sq_maxrank_HNF())
    assert C.matrix == B.matrix[:, 2:]
    assert C.mult_tab() == {0: {0: [-2, -2], 1: [0, 0]}, 1: {1: [0, 0]}}