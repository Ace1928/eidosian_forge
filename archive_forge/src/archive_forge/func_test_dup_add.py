from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_add():
    assert dup_add([], [], ZZ) == []
    assert dup_add([ZZ(1)], [], ZZ) == [ZZ(1)]
    assert dup_add([], [ZZ(1)], ZZ) == [ZZ(1)]
    assert dup_add([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(2)]
    assert dup_add([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(3)]
    assert dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(3)]
    assert dup_add([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(3)]
    assert dup_add([ZZ(1), ZZ(2), ZZ(3)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(9), ZZ(11), ZZ(13)]
    assert dup_add([], [], QQ) == []
    assert dup_add([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    assert dup_add([], [QQ(1, 2)], QQ) == [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 4)], QQ) == [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 2)], QQ) == [QQ(3, 4)]
    assert dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ) == [QQ(1, 2), QQ(5, 3)]
    assert dup_add([QQ(1)], [QQ(1, 2), QQ(2, 3)], QQ) == [QQ(1, 2), QQ(5, 3)]
    assert dup_add([QQ(1, 7), QQ(2, 7), QQ(3, 7)], [QQ(8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(9, 7), QQ(11, 7), QQ(13, 7)]