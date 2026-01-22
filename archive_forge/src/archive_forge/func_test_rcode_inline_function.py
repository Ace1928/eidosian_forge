from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.sets import Range
from sympy.logic import ITE
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises
from sympy.printing.rcode import RCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.rcode import rcode
def test_rcode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2 * x))
    assert rcode(g(x)) == '2*x'
    g = implemented_function('g', Lambda(x, 2 * x / Catalan))
    assert rcode(g(x)) == 'Catalan = %s;\n2*x/Catalan' % Catalan.n()
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x * (1 + x) * (2 + x)))
    res = rcode(g(A[i]), assign_to=A[i])
    ref = 'for (i in 1:n){\n   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n}'
    assert res == ref