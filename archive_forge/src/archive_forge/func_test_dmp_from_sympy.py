from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_from_sympy():
    assert dmp_from_sympy([[S.One, S(2)], [S.Zero]], 1, ZZ) == [[ZZ(1), ZZ(2)], []]
    assert dmp_from_sympy([[S.Half, S(2)]], 1, QQ) == [[QQ(1, 2), QQ(2, 1)]]