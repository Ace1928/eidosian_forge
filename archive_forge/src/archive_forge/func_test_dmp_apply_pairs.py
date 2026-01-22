from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_apply_pairs():
    h = lambda a, b: a * b
    assert dmp_apply_pairs([1, 2, 3], [4, 5, 6], h, [], 0, ZZ) == [4, 10, 18]
    assert dmp_apply_pairs([2, 3], [4, 5, 6], h, [], 0, ZZ) == [10, 18]
    assert dmp_apply_pairs([1, 2, 3], [5, 6], h, [], 0, ZZ) == [10, 18]
    assert dmp_apply_pairs([[1, 2], [3]], [[4, 5], [6]], h, [], 1, ZZ) == [[4, 10], [18]]
    assert dmp_apply_pairs([[1, 2], [3]], [[4], [5, 6]], h, [], 1, ZZ) == [[8], [18]]
    assert dmp_apply_pairs([[1], [2, 3]], [[4, 5], [6]], h, [], 1, ZZ) == [[5], [18]]