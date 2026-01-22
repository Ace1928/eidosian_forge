from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_mul_term():
    assert dmp_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, 0, ZZ) == dup_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, ZZ)
    assert dmp_mul_term([[]], [ZZ(2)], 3, 1, ZZ) == [[]]
    assert dmp_mul_term([[ZZ(1)]], [], 3, 1, ZZ) == [[]]
    assert dmp_mul_term([[ZZ(1), ZZ(2)], [ZZ(3)]], [ZZ(2)], 2, 1, ZZ) == [[ZZ(2), ZZ(4)], [ZZ(6)], [], []]
    assert dmp_mul_term([[]], [QQ(2, 3)], 3, 1, QQ) == [[]]
    assert dmp_mul_term([[QQ(1, 2)]], [], 3, 1, QQ) == [[]]
    assert dmp_mul_term([[QQ(1, 5), QQ(2, 5)], [QQ(3, 5)]], [QQ(2, 3)], 2, 1, QQ) == [[QQ(2, 15), QQ(4, 15)], [QQ(6, 15)], [], []]