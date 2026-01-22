from copy import copy
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.core.containers import Dict
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices import SparseMatrix
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
from sympy.testing.pytest import raises
def test_ndim_array_converting():
    dense_array = ImmutableDenseNDimArray([1, 2, 3, 4], (2, 2))
    alist = dense_array.tolist()
    assert alist == [[1, 2], [3, 4]]
    matrix = dense_array.tomatrix()
    assert isinstance(matrix, Matrix)
    for i in range(len(dense_array)):
        assert dense_array[dense_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == dense_array.shape
    assert ImmutableDenseNDimArray(matrix) == dense_array
    assert ImmutableDenseNDimArray(matrix.as_immutable()) == dense_array
    assert ImmutableDenseNDimArray(matrix.as_mutable()) == dense_array
    sparse_array = ImmutableSparseNDimArray([1, 2, 3, 4], (2, 2))
    alist = sparse_array.tolist()
    assert alist == [[1, 2], [3, 4]]
    matrix = sparse_array.tomatrix()
    assert isinstance(matrix, SparseMatrix)
    for i in range(len(sparse_array)):
        assert sparse_array[sparse_array._get_tuple_index(i)] == matrix[i]
    assert matrix.shape == sparse_array.shape
    assert ImmutableSparseNDimArray(matrix) == sparse_array
    assert ImmutableSparseNDimArray(matrix.as_immutable()) == sparse_array
    assert ImmutableSparseNDimArray(matrix.as_mutable()) == sparse_array