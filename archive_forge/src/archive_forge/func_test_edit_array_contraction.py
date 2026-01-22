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
def test_edit_array_contraction():
    cg = _array_contraction(_array_tensor_product(A, B, C, D), (1, 2, 5))
    ecg = _EditArrayContraction(cg)
    assert ecg.to_array_contraction() == cg
    ecg.args_with_ind[1], ecg.args_with_ind[2] = (ecg.args_with_ind[2], ecg.args_with_ind[1])
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, B, D), (1, 3, 4))
    ci = ecg.get_new_contraction_index()
    new_arg = _ArgE(X)
    new_arg.indices = [ci, ci]
    ecg.args_with_ind.insert(2, new_arg)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, X, B, D), (1, 3, 6), (4, 5))
    assert ecg.get_contraction_indices() == [[1, 3, 6], [4, 5]]
    assert [[tuple(j) for j in i] for i in ecg.get_contraction_indices_to_ind_rel_pos()] == [[(0, 1), (1, 1), (3, 0)], [(2, 0), (2, 1)]]
    assert [list(i) for i in ecg.get_mapping_for_index(0)] == [[0, 1], [1, 1], [3, 0]]
    assert [list(i) for i in ecg.get_mapping_for_index(1)] == [[2, 0], [2, 1]]
    raises(ValueError, lambda: ecg.get_mapping_for_index(2))
    ecg.args_with_ind.pop(1)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 4), (2, 3))
    ecg.args_with_ind[0].indices[1] = ecg.args_with_ind[1].indices[0]
    ecg.args_with_ind[1].indices[1] = ecg.args_with_ind[2].indices[0]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 2), (3, 4))
    ecg.insert_after(ecg.args_with_ind[1], _ArgE(C))
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, C, B, D), (1, 2), (3, 6))