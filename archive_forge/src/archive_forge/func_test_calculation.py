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
def test_calculation():
    a = ImmutableDenseNDimArray([1] * 9, (3, 3))
    b = ImmutableDenseNDimArray([9] * 9, (3, 3))
    c = a + b
    for i in c:
        assert i == ImmutableDenseNDimArray([10, 10, 10])
    assert c == ImmutableDenseNDimArray([10] * 9, (3, 3))
    assert c == ImmutableSparseNDimArray([10] * 9, (3, 3))
    c = b - a
    for i in c:
        assert i == ImmutableDenseNDimArray([8, 8, 8])
    assert c == ImmutableDenseNDimArray([8] * 9, (3, 3))
    assert c == ImmutableSparseNDimArray([8] * 9, (3, 3))