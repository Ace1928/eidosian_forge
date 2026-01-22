from copy import copy
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices import SparseMatrix
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import MutableSparseNDimArray
from sympy.testing.pytest import raises
def test_sympify():
    from sympy.abc import x, y, z, t
    arr = MutableDenseNDimArray([[x, y], [1, z * t]])
    arr_other = sympify(arr)
    assert arr_other.shape == (2, 2)
    assert arr_other == arr