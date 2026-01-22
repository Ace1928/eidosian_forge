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
def test_dmp_ground_primitive():
    assert dmp_ground_primitive([[]], 1, ZZ) == (ZZ(0), [[]])
    assert dmp_ground_primitive(f_0, 2, ZZ) == (ZZ(1), f_0)
    assert dmp_ground_primitive(dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == (ZZ(2), f_0)
    assert dmp_ground_primitive(f_1, 2, ZZ) == (ZZ(1), f_1)
    assert dmp_ground_primitive(dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == (ZZ(3), f_1)
    assert dmp_ground_primitive(f_2, 2, ZZ) == (ZZ(1), f_2)
    assert dmp_ground_primitive(dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == (ZZ(4), f_2)
    assert dmp_ground_primitive(f_3, 2, ZZ) == (ZZ(1), f_3)
    assert dmp_ground_primitive(dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == (ZZ(5), f_3)
    assert dmp_ground_primitive(f_4, 2, ZZ) == (ZZ(1), f_4)
    assert dmp_ground_primitive(dmp_mul_ground(f_4, ZZ(6), 2, ZZ), 2, ZZ) == (ZZ(6), f_4)
    assert dmp_ground_primitive(f_5, 2, ZZ) == (ZZ(1), f_5)
    assert dmp_ground_primitive(dmp_mul_ground(f_5, ZZ(7), 2, ZZ), 2, ZZ) == (ZZ(7), f_5)
    assert dmp_ground_primitive(f_6, 3, ZZ) == (ZZ(1), f_6)
    assert dmp_ground_primitive(dmp_mul_ground(f_6, ZZ(8), 3, ZZ), 3, ZZ) == (ZZ(8), f_6)
    assert dmp_ground_primitive([[ZZ(2)]], 1, ZZ) == (ZZ(2), [[ZZ(1)]])
    assert dmp_ground_primitive([[QQ(2)]], 1, QQ) == (QQ(2), [[QQ(1)]])
    assert dmp_ground_primitive([[QQ(2, 3)], [QQ(4, 9)]], 1, QQ) == (QQ(2, 9), [[QQ(3)], [QQ(2)]])
    assert dmp_ground_primitive([[QQ(2, 3)], [QQ(4, 5)]], 1, QQ) == (QQ(2, 15), [[QQ(5)], [QQ(6)]])