import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_tuple_func1d(self):

    def sample_1d(x):
        return (x[1], x[0])
    res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
    assert_array_equal(res, np.array([[2, 1], [4, 3]]))