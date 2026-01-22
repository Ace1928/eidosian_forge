from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_PowerBasis_element_from_poly():
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    f = Poly(1 + 2 * x)
    g = Poly(x ** 4)
    h = Poly(0, x)
    assert A.element_from_poly(f).coeffs == [1, 2, 0, 0]
    assert A.element_from_poly(g).coeffs == [-1, -1, -1, -1]
    assert A.element_from_poly(h).coeffs == [0, 0, 0, 0]