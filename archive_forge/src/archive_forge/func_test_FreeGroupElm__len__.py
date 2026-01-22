from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm__len__():
    assert len(x ** 5 * y * x ** 2 * y ** (-4) * x) == 13
    assert len(x ** 17) == 17
    assert len(y ** 0) == 0