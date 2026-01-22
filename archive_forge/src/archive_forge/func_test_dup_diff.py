from sympy.polys.densebasic import (
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densetools import (
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ, EX
from sympy.polys.rings import ring
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x
from sympy.testing.pytest import raises
def test_dup_diff():
    assert dup_diff([], 1, ZZ) == []
    assert dup_diff([7], 1, ZZ) == []
    assert dup_diff([2, 7], 1, ZZ) == [2]
    assert dup_diff([1, 2, 1], 1, ZZ) == [2, 2]
    assert dup_diff([1, 2, 3, 4], 1, ZZ) == [3, 4, 3]
    assert dup_diff([1, -1, 0, 0, 2], 1, ZZ) == [4, -3, 0, 0]
    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], ZZ)
    assert dup_diff(f, 0, ZZ) == f
    assert dup_diff(f, 1, ZZ) == [170, 306, 448, -2415, 138, 380, 0, 0, 24, 3]
    assert dup_diff(f, 2, ZZ) == dup_diff(dup_diff(f, 1, ZZ), 1, ZZ)
    assert dup_diff(f, 3, ZZ) == dup_diff(dup_diff(dup_diff(f, 1, ZZ), 1, ZZ), 1, ZZ)
    K = FF(3)
    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], K)
    assert dup_diff(f, 1, K) == dup_normal([2, 0, 1, 0, 0, 2, 0, 0, 0, 0], K)
    assert dup_diff(f, 2, K) == dup_normal([1, 0, 0, 2, 0, 0, 0], K)
    assert dup_diff(f, 3, K) == dup_normal([], K)
    assert dup_diff(f, 0, K) == f
    assert dup_diff(f, 2, K) == dup_diff(dup_diff(f, 1, K), 1, K)
    assert dup_diff(f, 3, K) == dup_diff(dup_diff(dup_diff(f, 1, K), 1, K), 1, K)