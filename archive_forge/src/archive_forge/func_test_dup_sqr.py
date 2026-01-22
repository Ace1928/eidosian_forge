from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_sqr():
    assert dup_sqr([], ZZ) == []
    assert dup_sqr([ZZ(2)], ZZ) == [ZZ(4)]
    assert dup_sqr([ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(4), ZZ(4)]
    assert dup_sqr([], QQ) == []
    assert dup_sqr([QQ(2, 3)], QQ) == [QQ(4, 9)]
    assert dup_sqr([QQ(1, 3), QQ(2, 3)], QQ) == [QQ(1, 9), QQ(4, 9), QQ(4, 9)]
    f = dup_normal([2, 0, 0, 1, 7], ZZ)
    assert dup_sqr(f, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)
    K = FF(9)
    assert dup_sqr([K(3), K(4)], K) == [K(6), K(7)]