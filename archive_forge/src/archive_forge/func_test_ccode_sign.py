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
def test_ccode_sign():
    expr1, ref1 = (sign(x) * y, 'y*(((x) > 0) - ((x) < 0))')
    expr2, ref2 = (sign(cos(x)), '(((cos(x)) > 0) - ((cos(x)) < 0))')
    expr3, ref3 = (sign(2 * x + x ** 2) * x + x ** 2, 'pow(x, 2) + x*(((pow(x, 2) + 2*x) > 0) - ((pow(x, 2) + 2*x) < 0))')
    assert ccode(expr1) == ref1
    assert ccode(expr1, 'z') == 'z = %s;' % ref1
    assert ccode(expr2) == ref2
    assert ccode(expr3) == ref3