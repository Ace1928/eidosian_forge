from sympy.external.importtools import version_tuple
from sympy.external import import_module
from sympy.core.numbers import (Float, Integer, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import (Matrix, list2numpy, matrix2numpy, symarray)
from sympy.utilities.lambdify import lambdify
import sympy
import mpmath
from sympy.abc import x, y, z
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.testing.pytest import raises
def test_Matrix_sum():
    M = Matrix([[1, 2, 3], [x, y, x], [2 * y, -50, z * x]])
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 3, 4], [x, 5, 6], [x, y, z ** 2]])
    assert M + m == Matrix([[3, 5, 7], [2 * x, y + 5, x + 6], [2 * y + x, y - 50, z * x + z ** 2]])
    assert m + M == Matrix([[3, 5, 7], [2 * x, y + 5, x + 6], [2 * y + x, y - 50, z * x + z ** 2]])
    assert M + m == M.add(m)