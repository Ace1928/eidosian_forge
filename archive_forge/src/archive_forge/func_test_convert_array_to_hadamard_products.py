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
def test_convert_array_to_hadamard_products():
    expr = HadamardProduct(M, N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr
    expr = HadamardProduct(M, N) * P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr
    expr = Q * HadamardProduct(M, N) * P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr
    expr = Q * HadamardProduct(M, N.T) * P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr
    expr = HadamardProduct(M, N) * HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret
    expr = P.T * HadamardProduct(M, N) * HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret
    cg = _array_diagonal(_array_tensor_product(M, N, Q), (1, 3), (0, 2, 4))
    ret = convert_array_to_matrix(cg)
    expected = PermuteDims(_array_diagonal(_array_tensor_product(HadamardProduct(M.T, N.T), Q), (1, 2)), [1, 0, 2])
    assert expected == ret
    cg = _array_diagonal(_array_tensor_product(HadamardProduct(M, N), Q), (0, 2))
    ret = convert_array_to_matrix(cg)
    assert ret == cg
    expr = Trace(HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N.T))
    expr = Trace(A * HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M, N) * A)
    expr = Trace(HadamardProduct(A, M) * N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N) * A)
    cg = _array_diagonal(_array_tensor_product(M, N), (0, 1, 2, 3))
    ret = convert_array_to_matrix(cg)
    assert ret == cg
    cg = _array_diagonal(_array_tensor_product(A), (0, 1))
    ret = convert_array_to_matrix(cg)
    assert ret == cg
    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 2, 4), (1, 3, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, N, P)
    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 3, 4), (1, 2, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, P, N.T)
    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)