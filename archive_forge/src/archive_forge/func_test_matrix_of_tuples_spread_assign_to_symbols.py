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
def test_matrix_of_tuples_spread_assign_to_symbols():
    gl = glsl_code
    with warns_deprecated_sympy():
        expr = Matrix([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    assign_to = (symbols('a b'), symbols('c d'), symbols('e f'), symbols('g h'))
    assert gl(expr, assign_to) == textwrap.dedent('        a = 1;\n        b = 2;\n        c = 3;\n        d = 4;\n        e = 5;\n        f = 6;\n        g = 7;\n        h = 8;')