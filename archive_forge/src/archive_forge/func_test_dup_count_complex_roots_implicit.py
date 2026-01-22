from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_count_complex_roots_implicit():
    R, x = ring('x', ZZ)
    f = x ** 5 - x
    assert R.dup_count_complex_roots(f) == 5
    assert R.dup_count_complex_roots(f, sup=(0, 0)) == 3
    assert R.dup_count_complex_roots(f, inf=(0, 0)) == 3