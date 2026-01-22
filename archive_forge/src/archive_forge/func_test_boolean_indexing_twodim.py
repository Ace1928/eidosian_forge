import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_indexing_twodim(self):
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[True, False, True], [False, True, False], [True, False, True]])
    assert_equal(a[b], [1, 3, 5, 7, 9])
    assert_equal(a[b[1]], [[4, 5, 6]])
    assert_equal(a[b[0]], a[b[2]])
    a[b] = 0
    assert_equal(a, [[0, 2, 0], [4, 0, 6], [0, 8, 0]])