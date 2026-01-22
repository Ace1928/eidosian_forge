from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_matrix_named_matsym():
    myout1 = MatrixSymbol('myout1', 1, 3)
    e2 = Matrix([[x, 2 * y, pi * z]])
    name_expr = ('test', Equality(myout1, e2, evaluate=False))
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function myout1 = test(x, y, z)\n  myout1 = [x 2*y pi*z];\nend\n'
    assert source == expected