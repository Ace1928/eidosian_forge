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
def test_ccode_cse():
    a, b, c, d = symbols('a b c d')
    e = MatrixSymbol('e', 3, 1)
    name_expr = ('test', [Equality(e, Matrix([[a * b], [a * b + c * d], [a * b * c * d]]))])
    generator = CCodeGen(cse=True)
    result = codegen(name_expr, code_gen=generator, header=False, empty=False)
    source = result[0][1]
    expected = '#include "test.h"\n#include <math.h>\nvoid test(double a, double b, double c, double d, double *e) {\n   const double x0 = a*b;\n   const double x1 = c*d;\n   e[0] = x0;\n   e[1] = x0 + x1;\n   e[2] = x0*x1;\n}\n'
    assert source == expected