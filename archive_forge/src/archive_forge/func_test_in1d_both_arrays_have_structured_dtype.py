import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_both_arrays_have_structured_dtype(self):
    dt = np.dtype([('field1', int), ('field2', object)])
    ar1 = np.array([(1, None)], dtype=dt)
    ar2 = np.array([(1, None)] * 10, dtype=dt)
    expected = np.array([True])
    result = np.in1d(ar1, ar2)
    assert_array_equal(result, expected)