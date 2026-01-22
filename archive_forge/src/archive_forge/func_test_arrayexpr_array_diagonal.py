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
def test_arrayexpr_array_diagonal():
    cg = _array_diagonal(M, (1, 0))
    assert cg == _array_diagonal(M, (0, 1))
    cg = _array_diagonal(_array_tensor_product(M, N, P), (4, 1), (2, 0))
    assert cg == _array_diagonal(_array_tensor_product(M, N, P), (1, 4), (0, 2))
    cg = _array_diagonal(_array_tensor_product(M, N), (1, 2), (3,), allow_trivial_diags=True)
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(M, N), (1, 2)), [0, 2, 1])
    Ax = ArraySymbol('Ax', shape=(1, 2, 3, 4, 3, 5, 6, 2, 7))
    cg = _array_diagonal(Ax, (1, 7), (3,), (2, 4), (6,), allow_trivial_diags=True)
    assert cg == _permute_dims(_array_diagonal(Ax, (1, 7), (2, 4)), [0, 2, 4, 5, 1, 6, 3])
    cg = _array_diagonal(M, (0,), allow_trivial_diags=True)
    assert cg == _permute_dims(M, [1, 0])
    raises(ValueError, lambda: _array_diagonal(M, (0, 0)))