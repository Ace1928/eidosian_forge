from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm__mul__pow__():
    x1 = x.group.dtype(((Symbol('x'), 1),))
    assert x ** 2 == x1 * x
    assert (x ** 2 * y * x ** (-2)) ** 4 == x ** 2 * y ** 4 * x ** (-2)
    assert (x ** 2) ** 2 == x ** 4
    assert (x ** (-1)) ** (-1) == x
    assert (x ** (-1)) ** 0 == F.identity
    assert (y ** 2) ** (-2) == y ** (-4)
    assert x ** 2 * x ** (-1) == x
    assert x ** 2 * y ** 2 * y ** (-1) == x ** 2 * y
    assert x * x ** (-1) == F.identity
    assert x / x == F.identity
    assert x / x ** 2 == x ** (-1)
    assert x ** 2 * y / (x ** 2 * y ** (-1)) == x ** 2 * y ** 2 * x ** (-2)
    assert x ** 2 * y / (y ** (-1) * x ** 2) == x ** 2 * y * x ** (-2) * y
    assert x * (x ** (-1) * y * z * y ** (-1)) == y * z * y ** (-1)
    assert x ** 2 * (x ** (-2) * y ** (-1) * z ** 2 * y) == y ** (-1) * z ** 2 * y