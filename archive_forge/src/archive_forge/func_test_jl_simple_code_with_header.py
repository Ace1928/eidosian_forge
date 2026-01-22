from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_simple_code_with_header():
    name_expr = ('test', (x + y) * z)
    result, = codegen(name_expr, 'Julia', header=True, empty=False)
    assert result[0] == 'test.jl'
    source = result[1]
    expected = '#   Code generated with SymPy ' + sympy.__version__ + "\n#\n#   See http://www.sympy.org/ for more information.\n#\n#   This file is part of 'project'\nfunction test(x, y, z)\n    out1 = z .* (x + y)\n    return out1\nend\n"
    assert source == expected