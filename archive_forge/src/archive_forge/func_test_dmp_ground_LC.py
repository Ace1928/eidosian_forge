from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_ground_LC():
    assert dmp_ground_LC([[]], 1, ZZ) == 0
    assert dmp_ground_LC([[2, 3, 4], [5]], 1, ZZ) == 2
    assert dmp_ground_LC([[[]]], 2, ZZ) == 0
    assert dmp_ground_LC([[[2], [3, 4]], [[5]]], 2, ZZ) == 2