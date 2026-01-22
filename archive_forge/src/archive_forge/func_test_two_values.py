import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_two_values(self):
    """Test proper casting for two different values."""
    expected = np.array([[3, 4]] * 10)
    for x in ([3, 4], [[3, 4]]):
        result = _as_pairs(x, 10)
        assert_equal(result, expected)
    obj = object()
    assert_equal(_as_pairs(['a', obj], 10), np.array([['a', obj]] * 10))
    assert_equal(_as_pairs([[3], [4]], 2), np.array([[3, 3], [4, 4]]))
    assert_equal(_as_pairs([['a'], [obj]], 2), np.array([['a', 'a'], [obj, obj]]))