from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_true_LT():
    assert dmp_true_LT([[]], 1, ZZ) == ((0, 0), 0)
    assert dmp_true_LT([[7]], 1, ZZ) == ((0, 0), 7)
    assert dmp_true_LT([[1, 0]], 1, ZZ) == ((0, 1), 1)
    assert dmp_true_LT([[1], []], 1, ZZ) == ((1, 0), 1)
    assert dmp_true_LT([[1, 0], []], 1, ZZ) == ((1, 1), 1)