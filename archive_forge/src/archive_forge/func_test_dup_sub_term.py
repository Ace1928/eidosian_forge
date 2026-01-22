from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_sub_term():
    f = dup_normal([], ZZ)
    assert dup_sub_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)
    assert dup_sub_term(f, ZZ(1), 0, ZZ) == dup_normal([-1], ZZ)
    assert dup_sub_term(f, ZZ(1), 1, ZZ) == dup_normal([-1, 0], ZZ)
    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([-1, 0, 0], ZZ)
    f = dup_normal([1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(2), 0, ZZ) == dup_normal([1, 1, -1], ZZ)
    assert dup_sub_term(f, ZZ(2), 1, ZZ) == dup_normal([1, -1, 1], ZZ)
    assert dup_sub_term(f, ZZ(2), 2, ZZ) == dup_normal([-1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 3, ZZ) == dup_normal([-1, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 4, ZZ) == dup_normal([-1, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 5, ZZ) == dup_normal([-1, 0, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 6, ZZ) == dup_normal([-1, 0, 0, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 1], ZZ)