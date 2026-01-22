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
def test_no_square_ternary_quadratic():
    assert check_solutions(2 * x * y + y * z - 3 * x * z)
    assert check_solutions(189 * x * y - 345 * y * z - 12 * x * z)
    assert check_solutions(23 * x * y + 34 * y * z)
    assert check_solutions(x * y + y * z + z * x)
    assert check_solutions(23 * x * y + 23 * y * z + 23 * x * z)