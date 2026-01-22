from sympy.abc import x, zeta
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.exceptions import (
from sympy.polys.numberfields.modules import (
from sympy.polys.numberfields.utilities import is_int
from sympy.polys.polyerrors import UnificationFailed
from sympy.testing.pytest import raises
def test_Module_NotImplemented():
    M = Module()
    raises(NotImplementedError, lambda: M.n)
    raises(NotImplementedError, lambda: M.mult_tab())
    raises(NotImplementedError, lambda: M.represent(None))
    raises(NotImplementedError, lambda: M.starts_with_unity())
    raises(NotImplementedError, lambda: M.element_from_rational(QQ(2, 3)))