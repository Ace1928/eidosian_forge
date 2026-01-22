from sympy.core import (S, pi, oo, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix
from sympy.printing.rust import rust_code
def test_loops_addfactor():
    m, n, o, p = symbols('m n o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    code = rust_code((a[i, j, k, l] + b[i, j, k, l]) * c[j, k, l], assign_to=y[i])
    assert code == 'for i in 0..m {\n    y[i] = 0;\n}\nfor i in 0..m {\n    for j in 0..n {\n        for k in 0..o {\n            for l in 0..p {\n                y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i * n * o * p + j * o * p + k * p + l, i * n * o * p + j * o * p + k * p + l, j * o * p + k * p + l) + '            }\n        }\n    }\n}'