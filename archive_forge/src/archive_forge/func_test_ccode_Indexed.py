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
def test_ccode_Indexed():
    s, n, m, o = symbols('s n m o', integer=True)
    i, j, k = (Idx('i', n), Idx('j', m), Idx('k', o))
    x = IndexedBase('x')[j]
    A = IndexedBase('A')[i, j]
    B = IndexedBase('B')[i, j, k]
    p = C99CodePrinter()
    assert p._print_Indexed(x) == 'x[j]'
    assert p._print_Indexed(A) == 'A[%s]' % (m * i + j)
    assert p._print_Indexed(B) == 'B[%s]' % (i * o * m + j * o + k)
    A = IndexedBase('A', shape=(5, 3))[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (3 * i + j)
    A = IndexedBase('A', shape=(5, 3), strides='F')[i, j]
    assert ccode(A) == 'A[%s]' % (i + 5 * j)
    A = IndexedBase('A', shape=(29, 29), strides=(1, s), offset=o)[i, j]
    assert ccode(A) == 'A[o + s*j + i]'
    Abase = IndexedBase('A', strides=(s, m, n), offset=o)
    assert ccode(Abase[i, j, k]) == 'A[m*j + n*k + o + s*i]'
    assert ccode(Abase[2, 3, k]) == 'A[3*m + n*k + o + 2*s]'