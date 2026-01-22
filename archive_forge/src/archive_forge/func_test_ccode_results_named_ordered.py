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
def test_ccode_results_named_ordered():
    x, y, z = symbols('x,y,z')
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(A, Matrix([[1, 2, x]]))
    expr2 = Equality(C, (x + y) * z)
    expr3 = Equality(B, 2 * x)
    name_expr = ('test', [expr1, expr2, expr3])
    expected = '#include "test.h"\n#include <math.h>\nvoid test(double x, double *C, double z, double y, double *A, double *B) {\n   (*C) = z*(x + y);\n   A[0] = 1;\n   A[1] = 2;\n   A[2] = x;\n   (*B) = 2*x;\n}\n'
    result = codegen(name_expr, 'c', 'test', header=False, empty=False, argument_sequence=(x, C, z, y, A, B))
    source = result[0][1]
    assert source == expected