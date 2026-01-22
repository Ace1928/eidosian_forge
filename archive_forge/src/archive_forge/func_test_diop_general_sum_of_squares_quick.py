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
def test_diop_general_sum_of_squares_quick():
    for i in range(3, 10):
        assert check_solutions(sum((i ** 2 for i in symbols(':%i' % i))) - i)
    assert diop_general_sum_of_squares(x ** 2 + y ** 2 - 2) is None
    assert diop_general_sum_of_squares(x ** 2 + y ** 2 + z ** 2 + 2) == set()
    eq = x ** 2 + y ** 2 + z ** 2 - (1 + 4 + 9)
    assert diop_general_sum_of_squares(eq) == {(1, 2, 3)}
    eq = u ** 2 + v ** 2 + x ** 2 + y ** 2 + z ** 2 - 1313
    assert len(diop_general_sum_of_squares(eq, 3)) == 3
    var = symbols(':5') + (symbols('6', negative=True),)
    eq = Add(*[i ** 2 for i in var]) - 112
    base_soln = {(0, 1, 1, 5, 6, -7), (1, 1, 1, 3, 6, -8), (2, 3, 3, 4, 5, -7), (0, 1, 1, 1, 3, -10), (0, 0, 4, 4, 4, -8), (1, 2, 3, 3, 5, -8), (0, 1, 2, 3, 7, -7), (2, 2, 4, 4, 6, -6), (1, 1, 3, 4, 6, -7), (0, 2, 3, 3, 3, -9), (0, 0, 2, 2, 2, -10), (1, 1, 2, 3, 4, -9), (0, 1, 1, 2, 5, -9), (0, 0, 2, 6, 6, -6), (1, 3, 4, 5, 5, -6), (0, 2, 2, 2, 6, -8), (0, 3, 3, 3, 6, -7), (0, 2, 3, 5, 5, -7), (0, 1, 5, 5, 5, -6)}
    assert diophantine(eq) == base_soln
    assert len(diophantine(eq, permute=True)) == 196800
    assert diophantine(12 - x ** 2 - y ** 2 - z ** 2) == {(2, 2, 2)}
    eq = a ** 2 + b ** 2 + c ** 2 + d ** 2 - 4
    raises(NotImplementedError, lambda: classify_diop(-eq))