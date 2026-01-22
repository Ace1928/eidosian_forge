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
def test_multidim_c_argument_cse():
    A_sym = MatrixSymbol('A', 3, 3)
    b_sym = MatrixSymbol('b', 3, 1)
    A = Matrix(A_sym)
    b = Matrix(b_sym)
    c = A * b
    cgen = CCodeGen(project='test', cse=True)
    r = cgen.routine('c', c)
    r.arguments[-1].result_var = 'out'
    r.arguments[-1]._name = 'out'
    code = get_string(cgen.dump_c, [r], prefix='test')
    expected = '#include "test.h"\n#include <math.h>\nvoid c(double *A, double *b, double *out) {\n   out[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];\n   out[1] = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];\n   out[2] = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];\n}\n'
    assert code == expected