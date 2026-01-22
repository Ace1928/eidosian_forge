from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
def test_find_DN():
    assert find_DN(x ** 2 - 2 * x - y ** 2) == (1, 1)
    assert find_DN(x ** 2 - 3 * y ** 2 - 5) == (3, 5)
    assert find_DN(x ** 2 - 2 * x * y - 4 * y ** 2 - 7) == (5, 7)
    assert find_DN(4 * x ** 2 - 8 * x * y - y ** 2 - 9) == (20, 36)
    assert find_DN(7 * x ** 2 - 2 * x * y - y ** 2 - 12) == (8, 84)
    assert find_DN(-3 * x ** 2 + 4 * x * y - y ** 2) == (1, 0)
    assert find_DN(-13 * x ** 2 - 7 * x * y + y ** 2 + 2 * x - 2 * y - 14) == (101, -7825480)