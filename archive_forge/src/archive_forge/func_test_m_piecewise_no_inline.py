from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
@XFAIL
def test_m_piecewise_no_inline():
    pw = Piecewise((0, x < -1), (x ** 2, x <= 1), (-x + 2, x > 1), (1, True))
    name_expr = ('pwtest', pw)
    result, = codegen(name_expr, 'Octave', header=False, empty=False, inline=False)
    source = result[1]
    expected = 'function out1 = pwtest(x)\n  if (x < -1)\n    out1 = 0;\n  elseif (x <= 1)\n    out1 = x.^2;\n  elseif (x > 1)\n    out1 = -x + 2;\n  else\n    out1 = 1;\n  end\nend\n'
    assert source == expected