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
def test_sum_of_three_squares():
    for i in [0, 1, 2, 34, 123, 34304595905, 34304595905394941, 343045959052344, 800, 801, 802, 803, 804, 805, 806]:
        a, b, c = sum_of_three_squares(i)
        assert a ** 2 + b ** 2 + c ** 2 == i
    assert sum_of_three_squares(7) is None
    assert sum_of_three_squares(4 ** 5 * 15) is None
    assert sum_of_three_squares(25) == (5, 0, 0)
    assert sum_of_three_squares(4) == (0, 0, 2)