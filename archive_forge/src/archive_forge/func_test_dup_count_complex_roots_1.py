from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_count_complex_roots_1():
    R, x = ring('x', ZZ)
    f = x - 1
    assert R.dup_count_complex_roots(f, a, b) == 1
    assert R.dup_count_complex_roots(f, c, d) == 1
    f = x + 1
    assert R.dup_count_complex_roots(f, a, b) == 1
    assert R.dup_count_complex_roots(f, c, d) == 0