from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_degree():
    assert dup_degree([]) is -oo
    assert dup_degree([1]) == 0
    assert dup_degree([1, 0]) == 1
    assert dup_degree([1, 0, 0, 0, 1]) == 4