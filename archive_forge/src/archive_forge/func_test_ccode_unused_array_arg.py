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
def test_ccode_unused_array_arg():
    x = MatrixSymbol('x', 2, 1)
    name_expr = ('test', 1.0)
    generator = CCodeGen()
    result = codegen(name_expr, code_gen=generator, header=False, empty=False, argument_sequence=(x,))
    source = result[0][1]
    expected = '#include "test.h"\n#include <math.h>\ndouble test(double *x) {\n   double test_result;\n   test_result = 1.0;\n   return test_result;\n}\n'
    assert source == expected