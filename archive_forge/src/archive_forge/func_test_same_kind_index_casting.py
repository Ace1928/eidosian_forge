import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_same_kind_index_casting(self):
    index = np.arange(5)
    u_index = index.astype(np.uintp)
    arr = np.arange(10)
    assert_array_equal(arr[index], arr[u_index])
    arr[u_index] = np.arange(5)
    assert_array_equal(arr, np.arange(10))
    arr = np.arange(10).reshape(5, 2)
    assert_array_equal(arr[index], arr[u_index])
    arr[u_index] = np.arange(5)[:, None]
    assert_array_equal(arr, np.arange(5)[:, None].repeat(2, axis=1))
    arr = np.arange(25).reshape(5, 5)
    assert_array_equal(arr[u_index, u_index], arr[index, index])