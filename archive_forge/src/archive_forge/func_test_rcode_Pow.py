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
def test_rcode_Pow():
    assert rcode(x ** 3) == 'x^3'
    assert rcode(x ** y ** 3) == 'x^(y^3)'
    g = implemented_function('g', Lambda(x, 2 * x))
    assert rcode(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == '(3.5*2*x)^(-x + y^x)/(x^2 + y)'
    assert rcode(x ** (-1.0)) == '1.0/x'
    assert rcode(x ** Rational(2, 3)) == 'x^(2.0/3.0)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, 'dpowi'), (lambda base, exp: not exp.is_integer, 'pow')]
    assert rcode(x ** 3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert rcode(x ** 3.2, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 3.2)'