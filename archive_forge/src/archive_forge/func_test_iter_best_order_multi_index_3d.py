import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_best_order_multi_index_3d():
    a = arange(12)
    i = nditer(a.reshape(2, 3, 2), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)])
    i = nditer(a.reshape(2, 3, 2)[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1)])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2, 0), (0, 2, 1), (0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 2, 0), (1, 2, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 1), (0, 0, 0), (0, 1, 1), (0, 1, 0), (0, 2, 1), (0, 2, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0), (1, 2, 1), (1, 2, 0)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0), (1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1), (1, 2, 1), (0, 2, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0), (1, 0, 0), (0, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 1), (0, 0, 1), (1, 0, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0)])