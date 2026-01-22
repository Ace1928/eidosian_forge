from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Submodule_reduce_element():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    B = A.whole_submodule()
    b = B(to_col([90, 84, 80, 75]), denom=120)
    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
    b_bar_expected = B(to_col([30, 24, 20, 15]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected
    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=4)
    b_bar_expected = B(to_col([0, 24, 20, 15]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected
    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=8)
    b_bar_expected = B(to_col([0, 9, 5, 0]), denom=120)
    b_bar = C.reduce_element(b)
    assert b_bar == b_bar_expected
    a = A(to_col([1, 2, 3, 4]))
    raises(NotImplementedError, lambda: C.reduce_element(a))
    C = B.submodule_from_matrix(DomainMatrix([[5, 4, 3, 2], [0, 8, 7, 6], [0, 0, 11, 12], [0, 0, 0, 1]], (4, 4), ZZ).transpose())
    raises(StructureError, lambda: C.reduce_element(b))