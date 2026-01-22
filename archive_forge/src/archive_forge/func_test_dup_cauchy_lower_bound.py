from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_cauchy_lower_bound():
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([], QQ))
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([QQ(1)], QQ))
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([QQ(1), QQ(0), QQ(0)], QQ))
    raises(DomainError, lambda: dup_cauchy_lower_bound([ZZ_I(1), ZZ_I(1)], ZZ_I))
    assert dup_cauchy_lower_bound([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(2, 3)