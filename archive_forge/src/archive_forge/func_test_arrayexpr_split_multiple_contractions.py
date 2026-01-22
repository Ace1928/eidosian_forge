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
def test_arrayexpr_split_multiple_contractions():
    a = MatrixSymbol('a', k, 1)
    b = MatrixSymbol('b', k, 1)
    A = MatrixSymbol('A', k, k)
    B = MatrixSymbol('B', k, k)
    C = MatrixSymbol('C', k, k)
    X = MatrixSymbol('X', k, k)
    cg = _array_contraction(_array_tensor_product(A.T, a, b, b.T, (A * X * b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    expected = _array_contraction(_array_tensor_product(A.T, DiagMatrix(a), OneArray(1), b, b.T, (A * X * b).applyfunc(cos)), (1, 3), (2, 9), (6, 7, 10))
    assert cg.split_multiple_contractions().dummy_eq(expected)
    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8), (3, 7))
    assert cg.split_multiple_contractions() == cg
    cg = _array_contraction(_array_tensor_product(a, b, A), (0, 2, 4), (1, 3))
    assert cg.split_multiple_contractions() == cg