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
def test_simple_c_code():
    x, y, z = symbols('x,y,z')
    expr = (x + y) * z
    routine = make_routine('test', expr)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = '#include "file.h"\n#include <math.h>\ndouble test(double x, double y, double z) {\n   double test_result;\n   test_result = z*(x + y);\n   return test_result;\n}\n'
    assert source == expected