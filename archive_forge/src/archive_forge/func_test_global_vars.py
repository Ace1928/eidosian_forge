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
def test_global_vars():
    x, y, z, t = symbols('x y z t')
    result = codegen(('f', x * y), 'F95', header=False, empty=False, global_vars=(y,))
    source = result[0][1]
    expected = 'REAL*8 function f(x)\nimplicit none\nREAL*8, intent(in) :: x\nf = x*y\nend function\n'
    assert source == expected
    expected = '#include "f.h"\n#include <math.h>\ndouble f(double x, double y) {\n   double f_result;\n   f_result = x*y + z;\n   return f_result;\n}\n'
    result = codegen(('f', x * y + z), 'C', header=False, empty=False, global_vars=(z, t))
    source = result[0][1]
    assert source == expected