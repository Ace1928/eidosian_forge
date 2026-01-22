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
def test_glsl_code_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)
    s = 'for (int i=0; i<m; i++){\n   y[i] = x[i] + z[i];\n}\nfor (int i=0; i<m; i++){\n   for (int j=0; j<n; j++){\n      y[i] = A[n*i + j]*x[j] + y[i];\n   }\n}'
    c = glsl_code(A[i, j] * x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s