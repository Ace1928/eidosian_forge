from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_inflate():
    assert dmp_inflate([1], (3,), 0, ZZ) == [1]
    assert dmp_inflate([[]], (3, 7), 1, ZZ) == [[]]
    assert dmp_inflate([[2]], (1, 2), 1, ZZ) == [[2]]
    assert dmp_inflate([[2, 0]], (1, 1), 1, ZZ) == [[2, 0]]
    assert dmp_inflate([[2, 0]], (1, 2), 1, ZZ) == [[2, 0, 0]]
    assert dmp_inflate([[2, 0]], (1, 3), 1, ZZ) == [[2, 0, 0, 0]]
    assert dmp_inflate([[1, 0, 0], [1], [1, 0]], (2, 1), 1, ZZ) == [[1, 0, 0], [], [1], [], [1, 0]]
    raises(IndexError, lambda: dmp_inflate([[]], (-3, 7), 1, ZZ))