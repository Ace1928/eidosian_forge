from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_compat_col():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    col = to_col([1, 2, 3, 4])
    row = col.transpose()
    assert A.is_compat_col(col) is True
    assert A.is_compat_col(row) is False
    assert A.is_compat_col(1) is False
    assert A.is_compat_col(DomainMatrix.eye(3, ZZ)[:, 0]) is False
    assert A.is_compat_col(DomainMatrix.eye(4, QQ)[:, 0]) is False
    assert A.is_compat_col(DomainMatrix.eye(4, ZZ)[:, 0]) is True