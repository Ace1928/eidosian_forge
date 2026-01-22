from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_sqr():
    assert dmp_sqr([ZZ(1), ZZ(2)], 0, ZZ) == dup_sqr([ZZ(1), ZZ(2)], ZZ)
    assert dmp_sqr([[[]]], 2, ZZ) == [[[]]]
    assert dmp_sqr([[[ZZ(2)]]], 2, ZZ) == [[[ZZ(4)]]]
    assert dmp_sqr([[[]]], 2, QQ) == [[[]]]
    assert dmp_sqr([[[QQ(2, 3)]]], 2, QQ) == [[[QQ(4, 9)]]]
    K = FF(9)
    assert dmp_sqr([[K(3)], [K(4)]], 1, K) == [[K(6)], [K(7)]]