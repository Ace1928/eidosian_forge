import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_large_int(self):
    data = np.array([2 ** 60, 2 ** 60 + 1], dtype=np.uint64)
    r = rankdata(data)
    assert_array_equal(r, [1.0, 2.0])
    data = np.array([2 ** 60, 2 ** 60 + 1], dtype=np.int64)
    r = rankdata(data)
    assert_array_equal(r, [1.0, 2.0])
    data = np.array([2 ** 60, -2 ** 60 + 1], dtype=np.int64)
    r = rankdata(data)
    assert_array_equal(r, [2.0, 1.0])