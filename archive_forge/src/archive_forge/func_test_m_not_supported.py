from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_not_supported():
    f = Function('f')
    name_expr = ('test', [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function [out1, out2] = test(x)\n  % unsupported: Derivative(f(x), x)\n  % unsupported: zoo\n  out1 = Derivative(f(x), x);\n  out2 = zoo;\nend\n'
    assert source == expected