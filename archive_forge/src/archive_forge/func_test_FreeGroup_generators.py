from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroup_generators():
    assert (x ** 2 * y ** 4 * z ** (-1)).contains_generators() == {x, y, z}
    assert (x ** (-1) * y ** 3).contains_generators() == {x, y}