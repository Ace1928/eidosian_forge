from sympy import Lambda, S, Dummy, KroneckerProduct
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
from sympy.matrices.expressions.special import (Identity, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
from sympy.matrices import Trace, MatMul, Transpose
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
from sympy.testing.pytest import raises
def test_arrayexpr_convert_array_to_matrix_support_function():
    assert _support_function_tp1_recognize([], [2 * k]) == 2 * k
    assert _support_function_tp1_recognize([(1, 2)], [A, 2 * k, B, 3]) == 6 * k * A * B
    assert _support_function_tp1_recognize([(0, 3), (1, 2)], [A, B]) == Trace(A * B)
    assert _support_function_tp1_recognize([(1, 2)], [A, B]) == A * B
    assert _support_function_tp1_recognize([(0, 2)], [A, B]) == A.T * B
    assert _support_function_tp1_recognize([(1, 3)], [A, B]) == A * B.T
    assert _support_function_tp1_recognize([(0, 3)], [A, B]) == A.T * B.T
    assert _support_function_tp1_recognize([(1, 2), (5, 6)], [A, B, C, D]) == _array_tensor_product(A * B, C * D)
    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(_array_tensor_product(A * C, B * D), [0, 2, 1, 3])
    assert _support_function_tp1_recognize([(0, 3), (1, 4)], [A, B, C]) == B * A * C
    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4), (7, 8)], [X, Y, A, B, C, D]) == X * Y * A * B * C * D
    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4)], [X, Y, A, B, C, D]) == _array_tensor_product(X * Y * A * B, C * D)
    assert _support_function_tp1_recognize([(1, 7), (3, 8), (4, 11)], [X, Y, A, B, C, D]) == PermuteDims(_array_tensor_product(X * B.T, Y * C, A.T * D.T), [0, 2, 4, 1, 3, 5])
    assert _support_function_tp1_recognize([(0, 1), (3, 6), (5, 8)], [X, A, B, C, D]) == PermuteDims(_array_tensor_product(Trace(X) * A * C, B * D), [0, 2, 1, 3])
    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [A, A, B, C, D]) == A ** 2 * B * C * D
    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [X, A, B, C, D]) == X * A * B * C * D
    assert _support_function_tp1_recognize([(1, 6), (3, 8), (5, 10)], [X, Y, A, B, C, D]) == PermuteDims(_array_tensor_product(X * B, Y * C, A * D), [0, 2, 4, 1, 3, 5])
    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(_array_tensor_product(A * C, B * D), [0, 2, 1, 3])
    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C * X.T * B * A * D
    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C * X.T * B * A * D