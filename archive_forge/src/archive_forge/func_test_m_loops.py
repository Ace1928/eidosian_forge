from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_loops():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    result, = codegen(('mat_vec_mult', Eq(y[i], A[i, j] * x[j])), 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function y = mat_vec_mult(A, m, n, x)\n  for i = 1:m\n    y(i) = 0;\n  end\n  for i = 1:m\n    for j = 1:n\n      y(i) = %(rhs)s + y(i);\n    end\n  end\nend\n'
    assert source == expected % {'rhs': 'A(%s, %s).*x(j)' % (i, j)} or source == expected % {'rhs': 'x(j).*A(%s, %s)' % (i, j)}