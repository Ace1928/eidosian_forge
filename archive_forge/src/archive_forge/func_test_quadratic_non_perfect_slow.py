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
@slow
def test_quadratic_non_perfect_slow():
    assert check_solutions(8 * x ** 2 + 10 * x * y - 2 * y ** 2 - 32 * x - 13 * y - 23)
    assert check_solutions(-3 * x ** 2 - 2 * x * y + 7 * y ** 2 - 5 * x - 7)
    assert check_solutions(-4 - x + 4 * x ** 2 - y - 3 * x * y - 4 * y ** 2)
    assert check_solutions(1 + 2 * x + 2 * x ** 2 + 2 * y + x * y - 2 * y ** 2)