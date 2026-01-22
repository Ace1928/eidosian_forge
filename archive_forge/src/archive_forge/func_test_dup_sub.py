from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_sub():
    assert dup_sub([], [], ZZ) == []
    assert dup_sub([ZZ(1)], [], ZZ) == [ZZ(1)]
    assert dup_sub([], [ZZ(1)], ZZ) == [ZZ(-1)]
    assert dup_sub([ZZ(1)], [ZZ(1)], ZZ) == []
    assert dup_sub([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(-1)]
    assert dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(1)]
    assert dup_sub([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(-1), ZZ(-1)]
    assert dup_sub([ZZ(3), ZZ(2), ZZ(1)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(-5), ZZ(-7), ZZ(-9)]
    assert dup_sub([], [], QQ) == []
    assert dup_sub([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    assert dup_sub([], [QQ(1, 2)], QQ) == [QQ(-1, 2)]
    assert dup_sub([QQ(1, 3)], [QQ(1, 3)], QQ) == []
    assert dup_sub([QQ(1, 3)], [QQ(2, 3)], QQ) == [QQ(-1, 3)]
    assert dup_sub([QQ(1, 7), QQ(2, 7)], [QQ(1)], QQ) == [QQ(1, 7), QQ(-5, 7)]
    assert dup_sub([QQ(1)], [QQ(1, 7), QQ(2, 7)], QQ) == [QQ(-1, 7), QQ(5, 7)]
    assert dup_sub([QQ(3, 7), QQ(2, 7), QQ(1, 7)], [QQ(8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(-5, 7), QQ(-7, 7), QQ(-9, 7)]