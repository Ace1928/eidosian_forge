from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_empty_m_code():
    code_gen = OctaveCodeGen()
    output = StringIO()
    code_gen.dump_m([], output, 'file', header=False, empty=False)
    source = output.getvalue()
    assert source == ''