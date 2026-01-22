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
def test_arrayexpr_convert_array_to_diagonalized_vector():
    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T
    cg = _array_tensor_product(I1, a, b)
    assert convert_array_to_matrix(cg) == a * b.T
    cg = _array_contraction(_array_tensor_product(A, B, C), (0, 3), (1, 2))
    assert convert_array_to_matrix(cg) == Trace(A * B) * C
    cg = _array_diagonal(_array_tensor_product(A, a), (1, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (1, 3))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a)
    cg = _array_diagonal(_array_tensor_product(a, b), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _permute_dims(_array_contraction(_array_tensor_product(DiagMatrix(a), OneArray(1), b), (0, 3)), [1, 2, 0])
    assert convert_array_to_matrix(cg) == b.T * DiagMatrix(a)
    cg = _array_diagonal(_array_tensor_product(A, a), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (0, 3))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a)
    cg = _array_diagonal(_array_tensor_product(I, x, I1), (0, 2), (3, 5))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(I, OneArray(1), I1, DiagMatrix(x)), (0, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)
    cg = _array_diagonal(_array_tensor_product(I, x, A, B), (1, 2), (5, 6))
    assert _array_diag2contr_diagmatrix(cg) == _array_diagonal(_array_contraction(_array_tensor_product(I, OneArray(1), A, B, DiagMatrix(x)), (1, 7)), (5, 6))
    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3, 5))
    assert convert_array_to_matrix(cg) == a * b.T
    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), a, b, I1), (2, 6))
    assert convert_array_to_matrix(cg) == a * b.T
    cg = _array_diagonal(_array_tensor_product(x, I1), (1, 2))
    assert isinstance(cg, ArrayDiagonal)
    assert cg.diagonal_indices == ((1, 2),)
    assert convert_array_to_matrix(cg) == x
    cg = _array_diagonal(_array_tensor_product(x, I), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), I, DiagMatrix(x)), (1, 3))
    assert convert_array_to_matrix(cg).doit() == DiagMatrix(x)
    raises(ValueError, lambda: _array_diagonal(x, (1,)))
    cg = _array_contraction(_array_tensor_product(I, A, I, I), (0, 2), (1, 3), (5, 7))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg) == Trace(A) * I
    cg = _array_contraction(_array_tensor_product(Trace(A) * I, I, I), (1, 5), (3, 4))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg).doit() == Trace(A) * I
    cg = _array_contraction(_array_tensor_product(A, a), (1, 2))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg) == A * a
    cg = _array_contraction(_array_tensor_product(A, a, B), (1, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * B
    cg = _array_contraction(_array_tensor_product(A, a, B), (0, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (0, 2), (3, 5))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * B
    cg = _array_contraction(_array_tensor_product(A, a, b, a.T, B), (0, 2, 4, 7, 9))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), DiagMatrix(b), OneArray(1), DiagMatrix(a), OneArray(1), B), (0, 2), (3, 5), (6, 9), (8, 12))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * DiagMatrix(b) * DiagMatrix(a) * B.T
    cg = _array_contraction(_array_tensor_product(I1, I1, I1), (1, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(I1, I1, OneArray(1), I1), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == 1
    cg = _array_contraction(_array_tensor_product(I, I, I, I, A), (1, 2, 8), (5, 6, 9))
    assert convert_array_to_matrix(cg.split_multiple_contractions()).doit() == A
    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8))
    expected = _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), C, DiagMatrix(a), OneArray(1), B), (1, 3), (2, 5), (6, 7), (8, 10))
    assert cg.split_multiple_contractions() == expected
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * C * DiagMatrix(a) * B
    cg = _array_contraction(_array_tensor_product(a, I1, b, I1, (a.T * b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    expected = _array_contraction(_array_tensor_product(a, I1, OneArray(1), b, I1, OneArray(1), (a.T * b).applyfunc(cos)), (1, 3), (2, 10), (6, 8), (7, 11))
    assert cg.split_multiple_contractions().dummy_eq(expected)
    assert convert_array_to_matrix(cg).doit().dummy_eq(MatMul(a, (a.T * b).applyfunc(cos), b.T))