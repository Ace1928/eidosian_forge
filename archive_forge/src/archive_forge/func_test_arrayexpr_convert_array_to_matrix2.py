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
def test_arrayexpr_convert_array_to_matrix2():
    cg = _array_contraction(_array_tensor_product(M, N), (1, 3))
    assert convert_array_to_matrix(cg) == M * N.T
    cg = PermuteDims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)
    cg = _array_tensor_product(M, PermuteDims(N, Permutation([1, 0])))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)
    cg = _array_contraction(PermuteDims(_array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)
    cg = _array_contraction(_array_tensor_product(M, N, P, PermuteDims(Q, Permutation([1, 0]))), (1, 5), (2, 3))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)
    cg = _array_tensor_product(M, PermuteDims(N, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)
    cg = _array_tensor_product(PermuteDims(M, [1, 0]), PermuteDims(N, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M.T, N.T)
    cg = _array_tensor_product(PermuteDims(N, [1, 0]), PermuteDims(M, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(N.T, M.T)
    cg = _array_contraction(M, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, k) * M * OneMatrix(k, 1)
    cg = _array_contraction(x, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, k) * x
    Xm = MatrixSymbol('Xm', m, n)
    cg = _array_contraction(Xm, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, m) * Xm * OneMatrix(n, 1)