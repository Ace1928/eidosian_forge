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
def test_glsl_code_Piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))
    p = glsl_code(expr)
    s = '((x < 1) ? (\n   x\n)\n: (\n   pow(x, 2.0)\n))'
    assert p == s
    assert glsl_code(expr, assign_to='c') == 'if (x < 1) {\n   c = x;\n}\nelse {\n   c = pow(x, 2.0);\n}'
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: glsl_code(expr))