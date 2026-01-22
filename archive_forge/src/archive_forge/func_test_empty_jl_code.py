from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_empty_jl_code():
    code_gen = JuliaCodeGen()
    output = StringIO()
    code_gen.dump_jl([], output, 'file', header=False, empty=False)
    source = output.getvalue()
    assert source == ''