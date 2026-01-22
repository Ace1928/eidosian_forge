from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_matrix_output_autoname_2():
    e1 = x + y
    e2 = Matrix([[2 * x, 2 * y, 2 * z]])
    e3 = Matrix([[x], [y], [z]])
    e4 = Matrix([[x, y], [z, 16]])
    name_expr = ('test', (e1, e2, e3, e4))
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function [out1, out2, out3, out4] = test(x, y, z)\n  out1 = x + y;\n  out2 = [2*x 2*y 2*z];\n  out3 = [x; y; z];\n  out4 = [x y; z 16];\nend\n'
    assert source == expected