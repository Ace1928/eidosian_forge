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
def test_arrayexpr_convert_array_contraction_tp_additions():
    a = ArrayAdd(_array_tensor_product(M, N), _array_tensor_product(N, M))
    tp = _array_tensor_product(P, a, Q)
    expr = _array_contraction(tp, (3, 4))
    expected = _array_tensor_product(P, ArrayAdd(_array_contraction(_array_tensor_product(M, N), (1, 2)), _array_contraction(_array_tensor_product(N, M), (1, 2))), Q)
    assert expr == expected
    assert convert_array_to_matrix(expr) == _array_tensor_product(P, M * N + N * M, Q)
    expr = _array_contraction(tp, (1, 2), (3, 4), (5, 6))
    result = _array_contraction(_array_tensor_product(P, ArrayAdd(_array_contraction(_array_tensor_product(M, N), (1, 2)), _array_contraction(_array_tensor_product(N, M), (1, 2))), Q), (1, 2), (3, 4))
    assert expr == result
    assert convert_array_to_matrix(expr) == P * (M * N + N * M) * Q