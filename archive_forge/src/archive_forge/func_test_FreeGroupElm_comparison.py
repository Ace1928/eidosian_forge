from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_comparison():
    assert not x * y == y * x
    assert x ** 0 == y ** 0
    assert x ** 2 < y ** 3
    assert not x ** 3 < y ** 2
    assert x * y < x ** 2 * y
    assert x ** 2 * y ** 2 < y ** 4
    assert not y ** 4 < y ** (-4)
    assert not y ** 4 < x ** (-4)
    assert y ** (-2) < y ** 2
    assert x ** 2 <= y ** 2
    assert x ** 2 <= x ** 2
    assert not y * z > z * y
    assert x > x ** (-1)
    assert not x ** 2 >= y ** 2