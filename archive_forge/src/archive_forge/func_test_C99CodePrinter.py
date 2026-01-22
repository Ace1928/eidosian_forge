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
def test_C99CodePrinter():
    assert C99CodePrinter().doprint(expm1(x)) == 'expm1(x)'
    assert C99CodePrinter().doprint(log1p(x)) == 'log1p(x)'
    assert C99CodePrinter().doprint(exp2(x)) == 'exp2(x)'
    assert C99CodePrinter().doprint(log2(x)) == 'log2(x)'
    assert C99CodePrinter().doprint(fma(x, y, -z)) == 'fma(x, y, -z)'
    assert C99CodePrinter().doprint(log10(x)) == 'log10(x)'
    assert C99CodePrinter().doprint(Cbrt(x)) == 'cbrt(x)'
    assert C99CodePrinter().doprint(hypot(x, y)) == 'hypot(x, y)'
    assert C99CodePrinter().doprint(loggamma(x)) == 'lgamma(x)'
    assert C99CodePrinter().doprint(Max(x, 3, x ** 2)) == 'fmax(3, fmax(x, pow(x, 2)))'
    assert C99CodePrinter().doprint(Min(x, 3)) == 'fmin(3, x)'
    c99printer = C99CodePrinter()
    assert c99printer.language == 'C'
    assert c99printer.standard == 'C99'
    assert 'restrict' in c99printer.reserved_words
    assert 'using' not in c99printer.reserved_words