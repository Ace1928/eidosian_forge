from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_count_complex_roots_8():
    R, x = ring('x', ZZ)
    f = x ** 9 + 3 * x ** 5 - 4 * x
    assert R.dup_count_complex_roots(f, a, b) == 9
    assert R.dup_count_complex_roots(f, c, d) == 4
    f = x ** 11 - 2 * x ** 9 + 3 * x ** 7 - 6 * x ** 5 - 4 * x ** 3 + 8 * x
    assert R.dup_count_complex_roots(f, a, b) == 9
    assert R.dup_count_complex_roots(f, c, d) == 4