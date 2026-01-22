from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_PowerBasis_mult_tab():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    M = A.mult_tab()
    exp = {0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}, 1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1], 3: [-1, -1, -1, -1]}, 2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]}, 3: {3: [0, 1, 0, 0]}}
    assert M == exp
    assert all((is_int(c) for u in M for v in M[u] for c in M[u][v]))