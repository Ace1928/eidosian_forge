import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_best_order_c_index_3d():
    a = arange(12)
    i = nditer(a.reshape(2, 3, 2), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    i = nditer(a.reshape(2, 3, 2)[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [6, 0, 8, 2, 10, 4, 7, 1, 9, 3, 11, 5])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 10, 2, 8, 0, 6, 5, 11, 3, 9, 1, 7])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])