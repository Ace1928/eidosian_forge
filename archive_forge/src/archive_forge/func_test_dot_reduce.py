from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ..affines import (
from ..eulerangles import euler2mat
from ..orientations import aff2axcodes
def test_dot_reduce():
    with pytest.raises(TypeError):
        dot_reduce()
    assert dot_reduce(1) == 1
    assert dot_reduce(None) is None
    assert dot_reduce([1, 2, 3]) == [1, 2, 3]
    vec = [1, 2, 3]
    mat = np.arange(4, 13).reshape((3, 3))
    assert_array_equal(dot_reduce(vec, mat), np.dot(vec, mat))
    assert_array_equal(dot_reduce(mat, vec), np.dot(mat, vec))
    mat2 = np.arange(13, 22).reshape((3, 3))
    assert_array_equal(dot_reduce(mat2, vec, mat), mat2 @ (vec @ mat))
    assert_array_equal(dot_reduce(mat, vec, mat2), mat @ (vec @ mat2))