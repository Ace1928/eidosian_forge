from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy
def test_jl_multifcns_per_file_w_header():
    name_expr = [('foo', [2 * x, 3 * y]), ('bar', [y ** 2, 4 * y])]
    result = codegen(name_expr, 'Julia', header=True, empty=False)
    assert result[0][0] == 'foo.jl'
    source = result[0][1]
    expected = '#   Code generated with SymPy ' + sympy.__version__ + "\n#\n#   See http://www.sympy.org/ for more information.\n#\n#   This file is part of 'project'\nfunction foo(x, y)\n    out1 = 2 * x\n    out2 = 3 * y\n    return out1, out2\nend\nfunction bar(y)\n    out1 = y .^ 2\n    out2 = 4 * y\n    return out1, out2\nend\n"
    assert source == expected