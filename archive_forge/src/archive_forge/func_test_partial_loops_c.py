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
def test_partial_loops_c():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A', shape=(m, p))
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', (o, m - 5))
    j = Idx('j', n)
    (f1, code), (f2, interface) = codegen(('matrix_vector', Eq(y[i], A[i, j] * x[j])), 'C99', 'file', header=False, empty=False)
    assert f1 == 'file.c'
    expected = '#include "file.h"\n#include <math.h>\nvoid matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y) {\n   for (int i=o; i<%(upperi)s; i++){\n      y[i] = 0;\n   }\n   for (int i=o; i<%(upperi)s; i++){\n      for (int j=0; j<n; j++){\n         y[i] = %(rhs)s + y[i];\n      }\n   }\n}\n' % {'upperi': m - 4, 'rhs': '%(rhs)s'}
    assert code == expected % {'rhs': 'A[%s]*x[j]' % (i * p + j)} or code == expected % {'rhs': 'A[%s]*x[j]' % (j + i * p)} or code == expected % {'rhs': 'x[j]*A[%s]' % (i * p + j)} or (code == expected % {'rhs': 'x[j]*A[%s]' % (j + i * p)})
    assert f2 == 'file.h'
    assert interface == '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\nvoid matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y);\n#endif\n'