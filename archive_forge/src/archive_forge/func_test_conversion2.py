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
def test_conversion2():
    a = 2 * list2numpy([x ** 2, x])
    b = list2numpy([2 * x ** 2, 2 * x])
    assert (a == b).all()
    one = Rational(1)
    zero = Rational(0)
    X = list2numpy([one, zero, zero])
    Y = one * X
    X = list2numpy([Symbol('a') + Rational(1, 2)])
    Y = X + X
    assert Y == array([1 + 2 * Symbol('a')])
    Y = Y + 1
    assert Y == array([2 + 2 * Symbol('a')])
    Y = X - X
    assert Y == array([0])