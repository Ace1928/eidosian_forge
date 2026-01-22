import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_average_matrix():
    y = np.matrix(np.random.rand(5, 5))
    assert_array_equal(y.mean(0), np.average(y, 0))
    a = np.matrix([[1, 2], [3, 4]])
    w = np.matrix([[1, 2], [3, 4]])
    r = np.average(a, axis=0, weights=w)
    assert_equal(type(r), np.matrix)
    assert_equal(r, [[2.5, 10.0 / 3]])