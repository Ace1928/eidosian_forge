from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.tensor.array import Array
from sympy.tensor.array.dense_ndim_array import (
from sympy.tensor.array.sparse_ndim_array import (
from sympy.abc import x, y
def test_issue_and_18715():
    for array_type in mutable_array_types:
        A = array_type([0, 1, 2])
        A[0] += 5
        assert A[0] == 5