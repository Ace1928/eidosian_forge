import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_best_order_f_index_2d():
    a = arange(6)
    i = nditer(a.reshape(2, 3), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 2, 4, 1, 3, 5])
    i = nditer(a.reshape(2, 3).copy(order='F'), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3)[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 3, 5, 0, 2, 4])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 2, 0, 5, 3, 1])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 3, 1, 4, 2, 0])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])