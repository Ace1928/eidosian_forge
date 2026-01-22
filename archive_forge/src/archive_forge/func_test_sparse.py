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
def test_sparse():
    sparse_array = ImmutableSparseNDimArray([0, 0, 0, 1], (2, 2))
    assert len(sparse_array) == 2 * 2
    assert len(sparse_array._sparse_array) == 1
    assert sparse_array.tolist() == [[0, 0], [0, 1]]
    for i, j in zip(sparse_array, [[0, 0], [0, 1]]):
        assert i == ImmutableSparseNDimArray(j)

    def sparse_assignment():
        sparse_array[0, 0] = 123
    assert len(sparse_array._sparse_array) == 1
    raises(TypeError, sparse_assignment)
    assert len(sparse_array._sparse_array) == 1
    assert sparse_array[0, 0] == 0
    assert sparse_array / 0 == ImmutableSparseNDimArray([[S.NaN, S.NaN], [S.NaN, S.ComplexInfinity]], (2, 2))
    assert ImmutableSparseNDimArray.zeros(100000, 200000) == ImmutableSparseNDimArray.zeros(100000, 200000)
    a = ImmutableSparseNDimArray({200001: 1}, (100000, 200000))
    assert a * 3 == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert 3 * a == ImmutableSparseNDimArray({200001: 3}, (100000, 200000))
    assert a * 0 == ImmutableSparseNDimArray({}, (100000, 200000))
    assert 0 * a == ImmutableSparseNDimArray({}, (100000, 200000))
    assert a / 3 == ImmutableSparseNDimArray({200001: Rational(1, 3)}, (100000, 200000))
    assert -a == ImmutableSparseNDimArray({200001: -1}, (100000, 200000))