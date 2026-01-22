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
def test_arrayexpr_permute_tensor_product():
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 1, 0, 5, 4, 6, 7]))
    cg2 = _array_tensor_product(N, _permute_dims(M, [1, 0]), _permute_dims(P, [1, 0]), Q)
    assert cg1 == cg2
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 5, 0, 1, 6, 7]))
    cg2 = _array_tensor_product(N, P, M, Q)
    assert cg1 == cg2
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 6, 5, 7, 0, 1]))
    assert cg1.expr == _array_tensor_product(N, P, Q, M)
    assert cg1.permutation == Permutation([0, 1, 2, 4, 3, 5, 6, 7])
    cg1 = _array_contraction(_permute_dims(_array_tensor_product(N, Q, Q, M), [2, 1, 5, 4, 0, 3, 6, 7]), [1, 2, 6])
    cg2 = _permute_dims(_array_contraction(_array_tensor_product(Q, Q, N, M), (3, 5, 6)), [0, 2, 3, 1, 4])
    assert cg1 == cg2
    cg1 = _array_contraction(_array_contraction(_array_contraction(_array_contraction(_permute_dims(_array_tensor_product(N, Q, Q, M), [2, 1, 5, 4, 0, 3, 6, 7]), [1, 2, 6]), [1, 3, 4]), [1]), [0])
    cg2 = _array_contraction(_array_tensor_product(M, N, Q, Q), (0, 3, 5), (1, 4, 7), (2,), (6,))
    assert cg1 == cg2