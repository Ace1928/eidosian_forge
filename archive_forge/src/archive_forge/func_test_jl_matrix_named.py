from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_matrix_named():
    e2 = Matrix([[x, 2 * y, pi * z]])
    name_expr = ('test', Equality(MatrixSymbol('myout1', 1, 3), e2))
    result = codegen(name_expr, 'Julia', header=False, empty=False)
    assert result[0][0] == 'test.jl'
    source = result[0][1]
    expected = 'function test(x, y, z)\n    myout1 = [x 2 * y pi * z]\n    return myout1\nend\n'
    assert source == expected