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
def test_dup_primitive():
    assert dup_primitive([], ZZ) == (ZZ(0), [])
    assert dup_primitive([ZZ(1)], ZZ) == (ZZ(1), [ZZ(1)])
    assert dup_primitive([ZZ(1), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(1)])
    assert dup_primitive([ZZ(2), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(1)])
    assert dup_primitive([ZZ(1), ZZ(2), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(2), ZZ(1)])
    assert dup_primitive([ZZ(2), ZZ(4), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(2), ZZ(1)])
    assert dup_primitive([], QQ) == (QQ(0), [])
    assert dup_primitive([QQ(1)], QQ) == (QQ(1), [QQ(1)])
    assert dup_primitive([QQ(1), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(1)])
    assert dup_primitive([QQ(2), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(1)])
    assert dup_primitive([QQ(1), QQ(2), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(2), QQ(1)])
    assert dup_primitive([QQ(2), QQ(4), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(2), QQ(1)])
    assert dup_primitive([QQ(2, 3), QQ(4, 9)], QQ) == (QQ(2, 9), [QQ(3), QQ(2)])
    assert dup_primitive([QQ(2, 3), QQ(4, 5)], QQ) == (QQ(2, 15), [QQ(5), QQ(6)])