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
def test_ccode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    s0 = 'for (int i=0; i<m; i++){\n   y[i] = 0;\n}\n'
    s1 = 'for (int i=0; i<m; i++){\n   for (int j=0; j<n; j++){\n      for (int k=0; k<o; k++){\n         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i * n * o + j * o + k) + '      }\n   }\n}\n'
    s2 = 'for (int i=0; i<m; i++){\n   for (int k=0; k<o; k++){\n      y[i] = a[%s]*b[k] + y[i];\n' % (i * o + k) + '   }\n}\n'
    s3 = 'for (int i=0; i<m; i++){\n   for (int j=0; j<n; j++){\n      y[i] = a[%s]*b[j] + y[i];\n' % (i * n + j) + '   }\n}\n'
    c = ccode(b[j] * a[i, j] + b[k] * a[i, k] + b[j] * b[k] * c[i, j, k], assign_to=y[i])
    assert c == s0 + s1 + s2 + s3[:-1] or c == s0 + s1 + s3 + s2[:-1] or c == s0 + s2 + s1 + s3[:-1] or (c == s0 + s2 + s3 + s1[:-1]) or (c == s0 + s3 + s1 + s2[:-1]) or (c == s0 + s3 + s2 + s1[:-1])