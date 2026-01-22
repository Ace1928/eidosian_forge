from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_methods():
    assert (x ** 0).order() == 1
    assert (y ** 2).order() is oo
    assert (x ** (-1) * y).commutator(x) == y ** (-1) * x ** (-1) * y * x
    assert len(x ** 2 * y ** (-1)) == 3
    assert len(x ** (-1) * y ** 3 * z) == 5