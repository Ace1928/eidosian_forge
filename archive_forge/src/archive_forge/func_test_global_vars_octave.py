from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_global_vars_octave():
    x, y, z, t = symbols('x y z t')
    result = codegen(('f', x * y), 'Octave', header=False, empty=False, global_vars=(y,))
    source = result[0][1]
    expected = 'function out1 = f(x)\n  global y\n  out1 = x.*y;\nend\n'
    assert source == expected
    result = codegen(('f', x * y + z), 'Octave', header=False, empty=False, argument_sequence=(x, y), global_vars=(z, t))
    source = result[0][1]
    expected = 'function out1 = f(x, y)\n  global t z\n  out1 = x.*y + z;\nend\n'
    assert source == expected