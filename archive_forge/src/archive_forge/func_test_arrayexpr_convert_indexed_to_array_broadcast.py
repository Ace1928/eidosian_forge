from sympy import tanh
from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import IndexedBase
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ArrayContraction, ArrayTensorProduct, \
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array, _convert_indexed_to_array
from sympy.testing.pytest import raises
def test_arrayexpr_convert_indexed_to_array_broadcast():
    A = ArraySymbol('A', (3, 3))
    B = ArraySymbol('B', (3, 3))
    expr = A[i, j] + B[k, l]
    O2 = OneArray(3, 3)
    expected = ArrayAdd(ArrayTensorProduct(A, O2), ArrayTensorProduct(O2, B))
    assert convert_indexed_to_array(expr) == expected
    assert convert_indexed_to_array(expr, [i, j, k, l]) == expected
    assert convert_indexed_to_array(expr, [l, k, i, j]) == ArrayAdd(PermuteDims(ArrayTensorProduct(O2, A), [1, 0, 2, 3]), PermuteDims(ArrayTensorProduct(B, O2), [1, 0, 2, 3]))
    expr = A[i, j] + B[j, k]
    O1 = OneArray(3)
    assert convert_indexed_to_array(expr, [i, j, k]) == ArrayAdd(ArrayTensorProduct(A, O1), ArrayTensorProduct(O1, B))
    C = ArraySymbol('C', (d0, d1))
    D = ArraySymbol('D', (d3, d1))
    expr = C[i, j] + D[k, j]
    assert convert_indexed_to_array(expr, [i, j, k]) == ArrayAdd(ArrayTensorProduct(C, OneArray(d3)), PermuteDims(ArrayTensorProduct(OneArray(d0), D), [0, 2, 1]))
    X = ArraySymbol('X', (5, 3))
    expr = X[i, n] - X[j, n]
    assert convert_indexed_to_array(expr, [i, j, n]) == ArrayAdd(ArrayTensorProduct(-1, OneArray(5), X), PermuteDims(ArrayTensorProduct(X, OneArray(5)), [0, 2, 1]))
    raises(ValueError, lambda: convert_indexed_to_array(C[i, j] + D[i, j]))