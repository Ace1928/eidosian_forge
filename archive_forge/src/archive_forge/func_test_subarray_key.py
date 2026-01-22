import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_subarray_key(self):
    a_dtype = np.dtype([('pos', int, 3), ('f', '<f4')])
    a = np.array([([1, 1, 1], np.pi), ([1, 2, 3], 0.0)], dtype=a_dtype)
    b_dtype = np.dtype([('pos', int, 3), ('g', '<f4')])
    b = np.array([([1, 1, 1], 3), ([3, 2, 1], 0.0)], dtype=b_dtype)
    expected_dtype = np.dtype([('pos', int, 3), ('f', '<f4'), ('g', '<f4')])
    expected = np.array([([1, 1, 1], np.pi, 3)], dtype=expected_dtype)
    res = join_by('pos', a, b)
    assert_equal(res.dtype, expected_dtype)
    assert_equal(res, expected)