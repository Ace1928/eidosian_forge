from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_eliminate_word():
    w = x ** 5 * y * x ** 2 * y ** (-4) * x
    assert w.eliminate_word(x, x ** 2) == x ** 10 * y * x ** 4 * y ** (-4) * x ** 2
    w3 = x ** 2 * y ** 3 * x ** (-1) * y
    assert w3.eliminate_word(x, x ** 2) == x ** 4 * y ** 3 * x ** (-2) * y
    assert w3.eliminate_word(x, y) == y ** 5
    assert w3.eliminate_word(x, y ** 4) == y ** 8
    assert w3.eliminate_word(y, x ** (-1)) == x ** (-3)
    assert w3.eliminate_word(x, y * z) == y * z * y * z * y ** 3 * z ** (-1)
    assert (y ** (-3)).eliminate_word(y, x ** (-1) * z ** (-1)) == z * x * z * x * z * x