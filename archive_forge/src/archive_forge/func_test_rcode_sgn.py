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
def test_rcode_sgn():
    expr = sign(x) * y
    assert rcode(expr) == 'y*sign(x)'
    p = rcode(expr, 'z')
    assert p == 'z = y*sign(x);'
    p = rcode(sign(2 * x + x ** 2) * x + x ** 2)
    assert p == 'x^2 + x*sign(x^2 + 2*x)'
    expr = sign(cos(x))
    p = rcode(expr)
    assert p == 'sign(cos(x))'