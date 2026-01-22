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
def test_dmp_integrate():
    assert dmp_integrate([[[]]], 1, 2, QQ) == [[[]]]
    assert dmp_integrate([[[]]], 2, 2, QQ) == [[[]]]
    assert dmp_integrate([[[QQ(1)]]], 1, 2, QQ) == [[[QQ(1)]], [[]]]
    assert dmp_integrate([[[QQ(1)]]], 2, 2, QQ) == [[[QQ(1, 2)]], [[]], [[]]]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 0, 1, QQ) == [[QQ(1)], [QQ(2)], [QQ(3)]]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 1, 1, QQ) == [[QQ(1, 3)], [QQ(1)], [QQ(3)], []]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 2, 1, QQ) == [[QQ(1, 12)], [QQ(1, 3)], [QQ(3, 2)], [], []]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 3, 1, QQ) == [[QQ(1, 60)], [QQ(1, 12)], [QQ(1, 2)], [], [], []]