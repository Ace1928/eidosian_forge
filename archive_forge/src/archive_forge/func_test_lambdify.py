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
@conserve_mpmath_dps
def test_lambdify():
    mpmath.mp.dps = 16
    sin02 = mpmath.mpf('0.198669330795061215459412627')
    f = lambdify(x, sin(x), 'numpy')
    prec = 1e-15
    assert -prec < f(0.2) - sin02 < prec
    if version_tuple(numpy.__version__) >= version_tuple('1.17'):
        with raises(TypeError):
            f(x)
    else:
        with raises(AttributeError):
            f(x)