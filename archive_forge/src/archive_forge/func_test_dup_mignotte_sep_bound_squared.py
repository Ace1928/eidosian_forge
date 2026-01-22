from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_mignotte_sep_bound_squared():
    raises(PolynomialError, lambda: dup_mignotte_sep_bound_squared([], QQ))
    raises(PolynomialError, lambda: dup_mignotte_sep_bound_squared([QQ(1)], QQ))
    assert dup_mignotte_sep_bound_squared([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(3, 5)