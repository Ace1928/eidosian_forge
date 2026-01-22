from io import StringIO
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
def test_c_fortran_omit_routine_name():
    x, y = symbols('x,y')
    name_expr = [('foo', 2 * x)]
    result = codegen(name_expr, 'F95', header=False, empty=False)
    expresult = codegen(name_expr, 'F95', 'foo', header=False, empty=False)
    assert result[0][1] == expresult[0][1]
    name_expr = ('foo', x * y)
    result = codegen(name_expr, 'F95', header=False, empty=False)
    expresult = codegen(name_expr, 'F95', 'foo', header=False, empty=False)
    assert result[0][1] == expresult[0][1]
    name_expr = ('foo', Matrix([[x, y], [x + y, x - y]]))
    result = codegen(name_expr, 'C89', header=False, empty=False)
    expresult = codegen(name_expr, 'C89', 'foo', header=False, empty=False)
    assert result[0][1] == expresult[0][1]