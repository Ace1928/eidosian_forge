from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_output_arg_mixed_unordered():
    from sympy.functions.elementary.trigonometric import cos, sin
    a = symbols('a')
    name_expr = ('foo', [cos(2 * x), Equality(y, sin(x)), cos(x), Equality(a, sin(2 * x))])
    result, = codegen(name_expr, 'Octave', header=False, empty=False)
    assert result[0] == 'foo.m'
    source = result[1]
    expected = 'function [out1, y, out3, a] = foo(x)\n  out1 = cos(2*x);\n  y = sin(x);\n  out3 = cos(x);\n  a = sin(2*x);\nend\n'
    assert source == expected