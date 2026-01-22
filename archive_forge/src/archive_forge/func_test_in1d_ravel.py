import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('kind', [None, 'sort', 'table'])
def test_in1d_ravel(self, kind):
    a = np.arange(6).reshape(2, 3)
    b = np.arange(3, 9).reshape(3, 2)
    long_b = np.arange(3, 63).reshape(30, 2)
    ec = np.array([False, False, False, True, True, True])
    assert_array_equal(in1d(a, b, assume_unique=True, kind=kind), ec)
    assert_array_equal(in1d(a, b, assume_unique=False, kind=kind), ec)
    assert_array_equal(in1d(a, long_b, assume_unique=True, kind=kind), ec)
    assert_array_equal(in1d(a, long_b, assume_unique=False, kind=kind), ec)