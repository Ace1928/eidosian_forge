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
def test_classify_diop():
    raises(TypeError, lambda: classify_diop(x ** 2 / 3 - 1))
    raises(ValueError, lambda: classify_diop(1))
    raises(NotImplementedError, lambda: classify_diop(w * x * y * z - 1))
    raises(NotImplementedError, lambda: classify_diop(x ** 3 + y ** 3 + z ** 4 - 90))
    assert classify_diop(14 * x ** 2 + 15 * x - 42) == ([x], {1: -42, x: 15, x ** 2: 14}, 'univariate')
    assert classify_diop(x * y + z) == ([x, y, z], {x * y: 1, z: 1}, 'inhomogeneous_ternary_quadratic')
    assert classify_diop(x * y + z + w + x ** 2) == ([w, x, y, z], {x * y: 1, w: 1, x ** 2: 1, z: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x * y + x * z + x ** 2 + 1) == ([x, y, z], {x * y: 1, x * z: 1, x ** 2: 1, 1: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x * y + z + w + 42) == ([w, x, y, z], {x * y: 1, w: 1, 1: 42, z: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x * y + z * w) == ([w, x, y, z], {x * y: 1, w * z: 1}, 'homogeneous_general_quadratic')
    assert classify_diop(x * y ** 2 + 1) == ([x, y], {x * y ** 2: 1, 1: 1}, 'cubic_thue')
    assert classify_diop(x ** 4 + y ** 4 + z ** 4 - (1 + 16 + 81)) == ([x, y, z], {1: -98, x ** 4: 1, z ** 4: 1, y ** 4: 1}, 'general_sum_of_even_powers')
    assert classify_diop(x ** 2 + y ** 2 + z ** 2) == ([x, y, z], {x ** 2: 1, y ** 2: 1, z ** 2: 1}, 'homogeneous_ternary_quadratic_normal')