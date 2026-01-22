import random
from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
from sympy.testing.pytest import raises
def test_array_expressions_no_canonicalization():
    tp = _array_tensor_product(M, N, P)
    expr = ArrayTensorProduct(tp, N)
    assert str(expr) == 'ArrayTensorProduct(ArrayTensorProduct(M, N, P), N)'
    assert expr.doit() == ArrayTensorProduct(M, N, P, N)
    expr = ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)
    assert str(expr) == 'ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)'
    assert expr.doit() == ArrayContraction(ArrayTensorProduct(M, N), (0, 1))
    expr = ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)
    assert str(expr) == 'ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)'
    assert expr.doit() == PermuteDims(ArrayDiagonal(ArrayTensorProduct(M, N), (0, 1)), [2, 0, 1])
    expr = ArrayTensorProduct(PermuteDims(M, [1, 0]), N)
    assert str(expr) == 'ArrayTensorProduct(PermuteDims(M, (0 1)), N)'
    assert expr.doit() == PermuteDims(ArrayTensorProduct(M, N), [1, 0, 2, 3])
    expr = ArrayContraction(_array_contraction(tp, (0, 2)), (0, 1))
    assert isinstance(expr, ArrayContraction)
    assert isinstance(expr.expr, ArrayContraction)
    assert str(expr) == 'ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))'
    assert expr.doit() == ArrayContraction(tp, (0, 2), (1, 3))
    expr = ArrayContraction(ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1)), (0, 1))
    assert expr.doit() == ArrayContraction(tp, (0, 1), (2, 3), (4, 5))
    expr = ArrayContraction(ArrayDiagonal(tp, (0, 1)), (0, 1))
    assert str(expr) == 'ArrayContraction(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))'
    assert expr.doit() == ArrayDiagonal(ArrayContraction(ArrayTensorProduct(N, M, P), (0, 1)), (0, 1))
    expr = ArrayContraction(PermuteDims(M, [1, 0]), (0, 1))
    assert str(expr) == 'ArrayContraction(PermuteDims(M, (0 1)), (0, 1))'
    assert expr.doit() == ArrayContraction(M, (0, 1))
    expr = ArrayDiagonal(ArrayDiagonal(tp, (0, 2)), (0, 1))
    assert str(expr) == 'ArrayDiagonal(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))'
    assert expr.doit() == ArrayDiagonal(tp, (0, 2), (1, 3))
    expr = ArrayDiagonal(ArrayDiagonal(ArrayDiagonal(tp, (0, 1)), (0, 1)), (0, 1))
    assert expr.doit() == ArrayDiagonal(tp, (0, 1), (2, 3), (4, 5))
    assert expr._canonicalize() == expr.doit()
    expr = ArrayDiagonal(ArrayContraction(tp, (0, 1)), (0, 1))
    assert str(expr) == 'ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))'
    assert expr.doit() == expr
    expr = ArrayDiagonal(PermuteDims(M, [1, 0]), (0, 1))
    assert str(expr) == 'ArrayDiagonal(PermuteDims(M, (0 1)), (0, 1))'
    assert expr.doit() == ArrayDiagonal(M, (0, 1))
    expr = ArrayAdd(M)
    assert isinstance(expr, ArrayAdd)
    assert expr.doit() == M
    expr = ArrayAdd(ArrayAdd(M, N), P)
    assert str(expr) == 'ArrayAdd(ArrayAdd(M, N), P)'
    assert expr.doit() == ArrayAdd(M, N, P)
    expr = ArrayAdd(M, ArrayAdd(N, ArrayAdd(P, M)))
    assert expr.doit() == ArrayAdd(M, N, P, M)
    assert expr._canonicalize() == ArrayAdd(M, N, ArrayAdd(P, M))
    expr = ArrayAdd(M, ZeroArray(k, k), N)
    assert str(expr) == 'ArrayAdd(M, ZeroArray(k, k), N)'
    assert expr.doit() == ArrayAdd(M, N)
    expr = PermuteDims(PermuteDims(M, [1, 0]), [1, 0])
    assert str(expr) == 'PermuteDims(PermuteDims(M, (0 1)), (0 1))'
    assert expr.doit() == M
    expr = PermuteDims(PermuteDims(PermuteDims(M, [1, 0]), [1, 0]), [1, 0])
    assert expr.doit() == PermuteDims(M, [1, 0])
    assert expr._canonicalize() == expr.doit()
    expr = Reshape(A, (k ** 2,))
    assert expr.shape == (k ** 2,)
    assert isinstance(expr, Reshape)