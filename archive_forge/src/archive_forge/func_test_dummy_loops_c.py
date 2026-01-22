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
def test_dummy_loops_c():
    from sympy.tensor import IndexedBase, Idx
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = '#include "file.h"\n#include <math.h>\nvoid test_dummies(int m_%(mno)i, double *x, double *y) {\n   for (int i_%(ino)i=0; i_%(ino)i<m_%(mno)i; i_%(ino)i++){\n      y[i_%(ino)i] = x[i_%(ino)i];\n   }\n}\n' % {'ino': i.label.dummy_index, 'mno': m.dummy_index}
    r = make_routine('test_dummies', Eq(y[i], x[i]))
    c89 = C89CodeGen()
    c99 = C99CodeGen()
    code = get_string(c99.dump_c, [r])
    assert code == expected
    with raises(NotImplementedError):
        get_string(c89.dump_c, [r])