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
def test_systematic_basic():

    def s(sympy_object, numpy_array):
        _ = [sympy_object + numpy_array, numpy_array + sympy_object, sympy_object - numpy_array, numpy_array - sympy_object, sympy_object * numpy_array, numpy_array * sympy_object, sympy_object / numpy_array, numpy_array / sympy_object, sympy_object ** numpy_array, numpy_array ** sympy_object]
    x = Symbol('x')
    y = Symbol('y')
    sympy_objs = [Rational(2, 3), Float('1.3'), x, y, pow(x, y) * y, Integer(5), Float(5.5)]
    numpy_objs = [array([1]), array([3, 8, -1]), array([x, x ** 2, Rational(5)]), array([x / y * sin(y), 5, Rational(5)])]
    for x in sympy_objs:
        for y in numpy_objs:
            s(x, y)