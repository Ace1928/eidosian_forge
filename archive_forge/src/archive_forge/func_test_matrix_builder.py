import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_matrix_builder(self):
    a = np.array([1])
    b = np.array([2])
    c = np.array([3])
    d = np.array([4])
    actual = np.r_['a, b; c, d']
    expected = np.bmat([[a, b], [c, d]])
    assert_equal(actual, expected)
    assert_equal(type(actual), type(expected))