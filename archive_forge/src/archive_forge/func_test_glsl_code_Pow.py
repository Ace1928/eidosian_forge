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
def test_glsl_code_Pow():
    g = implemented_function('g', Lambda(x, 2 * x))
    assert glsl_code(x ** 3) == 'pow(x, 3.0)'
    assert glsl_code(x ** y ** 3) == 'pow(x, pow(y, 3.0))'
    assert glsl_code(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == 'pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2.0) + y)'
    assert glsl_code(x ** (-1.0)) == '1.0/x'