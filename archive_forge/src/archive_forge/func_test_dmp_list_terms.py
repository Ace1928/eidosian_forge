from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_list_terms():
    assert dmp_list_terms([[[]]], 2, ZZ) == [((0, 0, 0), 0)]
    assert dmp_list_terms([[[1]]], 2, ZZ) == [((0, 0, 0), 1)]
    assert dmp_list_terms([1, 2, 4, 3, 5], 0, ZZ) == [((4,), 1), ((3,), 2), ((2,), 4), ((1,), 3), ((0,), 5)]
    assert dmp_list_terms([[1], [2, 4], [3, 5, 0]], 1, ZZ) == [((2, 0), 1), ((1, 1), 2), ((1, 0), 4), ((0, 2), 3), ((0, 1), 5)]
    f = [[2, 0, 0, 0], [1, 0, 0], []]
    assert dmp_list_terms(f, 1, ZZ, order='lex') == [((2, 3), 2), ((1, 2), 1)]
    assert dmp_list_terms(f, 1, ZZ, order='grlex') == [((2, 3), 2), ((1, 2), 1)]
    f = [[2, 0, 0, 0], [1, 0, 0, 0, 0, 0], []]
    assert dmp_list_terms(f, 1, ZZ, order='lex') == [((2, 3), 2), ((1, 5), 1)]
    assert dmp_list_terms(f, 1, ZZ, order='grlex') == [((1, 5), 1), ((2, 3), 2)]