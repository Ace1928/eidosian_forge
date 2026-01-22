import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('kind', [None, 'sort', 'table'])
def test_in1d_boolean(self, kind):
    """Test that in1d works for boolean input"""
    a = np.array([True, False])
    b = np.array([False, False, False])
    expected = np.array([False, True])
    assert_array_equal(expected, in1d(a, b, kind=kind))
    assert_array_equal(np.invert(expected), in1d(a, b, invert=True, kind=kind))