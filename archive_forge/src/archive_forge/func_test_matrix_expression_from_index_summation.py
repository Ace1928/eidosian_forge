from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises
def test_matrix_expression_from_index_summation():
    from sympy.abc import a, b, c, d
    A = MatrixSymbol('A', k, k)
    B = MatrixSymbol('B', k, k)
    C = MatrixSymbol('C', k, k)
    w1 = MatrixSymbol('w1', k, 1)
    i0, i1, i2, i3, i4 = symbols('i0:5', cls=Dummy)
    expr = Sum(W[a, b] * X[b, c] * Z[c, d], (b, 0, l - 1), (c, 0, m - 1))
    assert MatrixExpr.from_index_summation(expr, a) == W * X * Z
    expr = Sum(W.T[b, a] * X[b, c] * Z[c, d], (b, 0, l - 1), (c, 0, m - 1))
    assert MatrixExpr.from_index_summation(expr, a) == W * X * Z
    expr = Sum(A[b, a] * B[b, c] * C[c, d], (b, 0, k - 1), (c, 0, k - 1))
    assert MatrixSymbol.from_index_summation(expr, a) == A.T * B * C
    expr = Sum(A[b, a] * B[c, b] * C[c, d], (b, 0, k - 1), (c, 0, k - 1))
    assert MatrixSymbol.from_index_summation(expr, a) == A.T * B.T * C
    expr = Sum(C[c, d] * A[b, a] * B[c, b], (b, 0, k - 1), (c, 0, k - 1))
    assert MatrixSymbol.from_index_summation(expr, a) == A.T * B.T * C
    expr = Sum(A[a, b] + B[a, b], (a, 0, k - 1), (b, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == OneMatrix(1, k) * A * OneMatrix(k, 1) + OneMatrix(1, k) * B * OneMatrix(k, 1)
    expr = Sum(A[a, b] ** 2, (a, 0, k - 1), (b, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == Trace(A * A.T)
    expr = Sum(A[a, b] ** 3, (a, 0, k - 1), (b, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == Trace(HadamardPower(A.T, 2) * A)
    expr = Sum((A[a, b] + B[a, b]) * C[b, c], (b, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == (A + B) * C
    expr = Sum((A[a, b] + B[b, a]) * C[b, c], (b, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == (A + B.T) * C
    expr = Sum(A[a, b] * A[b, c] * A[c, d], (b, 0, k - 1), (c, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == A ** 3
    expr = Sum(A[a, b] * A[b, c] * B[c, d], (b, 0, k - 1), (c, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == A ** 2 * B
    expr = Sum(A[a, a], (a, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, None) == trace(A)
    expr = Sum(A[a, a] * B[b, c] * C[c, d], (a, 0, k - 1), (c, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, b) == trace(A) * B * C
    expr = Sum(W[a, b] * X[b, c] * Z[c, d], (b, 0, l - 1), (c, 0, m))
    raises(ValueError, lambda: MatrixExpr.from_index_summation(expr, a))
    expr = Sum(W[a, b] * X[b, c] * Z[c, d], (b, 0, l - 1), (c, 1, m - 1))
    raises(ValueError, lambda: MatrixExpr.from_index_summation(expr, a))
    expr = Sum(A[a, b] * Sum(B[b, c] * C[c, d], (c, 0, k - 1)), (b, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == A * B * C
    expr = Sum(A[a, b] * KroneckerDelta(b, c) * B[c, d], (b, 0, k - 1), (c, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, a) == A * B
    expr = Sum(KroneckerDelta(i1, m) * KroneckerDelta(i2, n) * A[i, i1] * A[j, i2], (i1, 0, k - 1), (i2, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, m) == ArrayTensorProduct(A.T, A)
    expr = Sum(A[i1, i2] * w1[i2, 0], (i2, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, i1) == MatrixElement(A * w1, i1, 0)
    expr = Sum(A[i1, i2] * B[i2, 0], (i2, 0, k - 1))
    assert MatrixExpr.from_index_summation(expr, i1) == MatrixElement(A * B, i1, 0)