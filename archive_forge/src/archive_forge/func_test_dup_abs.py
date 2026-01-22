from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_abs():
    assert dup_abs([], ZZ) == []
    assert dup_abs([ZZ(1)], ZZ) == [ZZ(1)]
    assert dup_abs([ZZ(-7)], ZZ) == [ZZ(7)]
    assert dup_abs([ZZ(-1), ZZ(2), ZZ(3)], ZZ) == [ZZ(1), ZZ(2), ZZ(3)]
    assert dup_abs([], QQ) == []
    assert dup_abs([QQ(1, 2)], QQ) == [QQ(1, 2)]
    assert dup_abs([QQ(-7, 3)], QQ) == [QQ(7, 3)]
    assert dup_abs([QQ(-1, 7), QQ(2, 7), QQ(3, 7)], QQ) == [QQ(1, 7), QQ(2, 7), QQ(3, 7)]