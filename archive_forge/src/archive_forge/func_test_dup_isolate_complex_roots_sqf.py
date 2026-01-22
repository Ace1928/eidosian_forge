from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_isolate_complex_roots_sqf():
    R, x = ring('x', ZZ)
    f = x ** 2 - 2 * x + 3
    assert R.dup_isolate_complex_roots_sqf(f) == [((0, -6), (6, 0)), ((0, 0), (6, 6))]
    assert [r.as_tuple() for r in R.dup_isolate_complex_roots_sqf(f, blackbox=True)] == [((0, -6), (6, 0)), ((0, 0), (6, 6))]
    assert R.dup_isolate_complex_roots_sqf(f, eps=QQ(1, 10)) == [((QQ(15, 16), -QQ(3, 2)), (QQ(33, 32), -QQ(45, 32))), ((QQ(15, 16), QQ(45, 32)), (QQ(33, 32), QQ(3, 2)))]
    assert R.dup_isolate_complex_roots_sqf(f, eps=QQ(1, 100)) == [((QQ(255, 256), -QQ(363, 256)), (QQ(513, 512), -QQ(723, 512))), ((QQ(255, 256), QQ(723, 512)), (QQ(513, 512), QQ(363, 256)))]
    f = 7 * x ** 4 - 19 * x ** 3 + 20 * x ** 2 + 17 * x + 20
    assert R.dup_isolate_complex_roots_sqf(f) == [((-QQ(40, 7), -QQ(40, 7)), (0, 0)), ((-QQ(40, 7), 0), (0, QQ(40, 7))), ((0, -QQ(40, 7)), (QQ(40, 7), 0)), ((0, 0), (QQ(40, 7), QQ(40, 7)))]