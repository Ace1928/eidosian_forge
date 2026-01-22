import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_broadcast_shape():
    assert_equal(_broadcast_shape(), ())
    assert_equal(_broadcast_shape([1, 2]), (2,))
    assert_equal(_broadcast_shape(np.ones((1, 1))), (1, 1))
    assert_equal(_broadcast_shape(np.ones((1, 1)), np.ones((3, 4))), (3, 4))
    assert_equal(_broadcast_shape(*[np.ones((1, 2))] * 32), (1, 2))
    assert_equal(_broadcast_shape(*[np.ones((1, 2))] * 100), (1, 2))
    assert_equal(_broadcast_shape(*[np.ones(2)] * 32 + [1]), (2,))
    bad_args = [np.ones(2)] * 32 + [np.ones(3)] * 32
    assert_raises(ValueError, lambda: _broadcast_shape(*bad_args))