from sympy.core import (
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.codeprinter import ccode
def test_ccode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {'ceiling': 'ceil', 'Abs': [(lambda x: not x.is_integer, 'fabs'), (lambda x: x.is_integer, 'abs')]}
    assert ccode(ceiling(x), user_functions=custom_functions) == 'ceil(x)'
    assert ccode(Abs(x), user_functions=custom_functions) == 'fabs(x)'
    assert ccode(Abs(n), user_functions=custom_functions) == 'abs(n)'
    expr = Symbol('a')
    muladd = Function('muladd')
    for i in range(0, 100):
        expr = muladd(Rational(1, 2), Symbol(f'a{i}'), expr)
    out = ccode(expr, user_functions={'muladd': 'muladd'})
    assert 'a99' in out
    assert out.count('muladd') == 100