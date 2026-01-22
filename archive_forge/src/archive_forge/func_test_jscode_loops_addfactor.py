from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    s = 'for (var i=0; i<m; i++){\n   y[i] = 0;\n}\nfor (var i=0; i<m; i++){\n   for (var j=0; j<n; j++){\n      for (var k=0; k<o; k++){\n         for (var l=0; l<p; l++){\n            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i * n * o * p + j * o * p + k * p + l, i * n * o * p + j * o * p + k * p + l, j * o * p + k * p + l) + '         }\n      }\n   }\n}'
    c = jscode((a[i, j, k, l] + b[i, j, k, l]) * c[j, k, l], assign_to=y[i])
    assert c == s