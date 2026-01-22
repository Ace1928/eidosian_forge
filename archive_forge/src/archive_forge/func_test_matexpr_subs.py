from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor
from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL
def test_matexpr_subs():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', m, l)
    assert A.subs(n, m).shape == (m, m)
    assert (A * B).subs(B, C) == A * C
    assert (A * B).subs(l, n).is_square
    W = MatrixSymbol('W', 3, 3)
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 1, 2)
    Z = MatrixSymbol('Z', n, 2)
    assert X.subs(X, Y) == Y
    y = Str('y')
    assert X.subs(Str('X'), y).args == (y, 2, 2)
    assert X[1, 1].subs(X, W) == W[1, 1]
    raises(IndexError, lambda: X[1, 1].subs(X, Y))
    assert X[0, 1].subs(X, Y) == Y[0, 1]
    assert W[2, 1].subs(W, Z) == Z[2, 1]
    raises(IndexError, lambda: W[2, 2].subs(W, Z))
    raises(IndexError, lambda: W[2, 2].subs(W, zeros(2)))
    A = SparseMatrix([[1, 2], [3, 4]])
    B = Matrix([[1, 2], [3, 4]])
    C, D = (MatrixSymbol('C', 2, 2), MatrixSymbol('D', 2, 2))
    assert (C * D).subs({C: A, D: B}) == MatMul(A, B)