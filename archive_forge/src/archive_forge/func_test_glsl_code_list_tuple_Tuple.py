from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing.glsl import GLSLPrinter
from sympy.printing.str import StrPrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.core import Tuple
from sympy.printing.glsl import glsl_code
import textwrap
def test_glsl_code_list_tuple_Tuple():
    assert glsl_code([1, 2, 3, 4]) == 'vec4(1, 2, 3, 4)'
    assert glsl_code([1, 2, 3], glsl_types=False) == 'float[3](1, 2, 3)'
    assert glsl_code([1, 2, 3]) == glsl_code((1, 2, 3))
    assert glsl_code([1, 2, 3]) == glsl_code(Tuple(1, 2, 3))
    m = MatrixSymbol('A', 3, 4)
    assert glsl_code([m[0], m[1]])