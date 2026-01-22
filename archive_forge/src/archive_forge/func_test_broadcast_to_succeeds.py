import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_broadcast_to_succeeds():
    data = [[np.array(0), (0,), np.array(0)], [np.array(0), (1,), np.zeros(1)], [np.array(0), (3,), np.zeros(3)], [np.ones(1), (1,), np.ones(1)], [np.ones(1), (2,), np.ones(2)], [np.ones(1), (1, 2, 3), np.ones((1, 2, 3))], [np.arange(3), (3,), np.arange(3)], [np.arange(3), (1, 3), np.arange(3).reshape(1, -1)], [np.arange(3), (2, 3), np.array([[0, 1, 2], [0, 1, 2]])], [np.ones(0), 0, np.ones(0)], [np.ones(1), 1, np.ones(1)], [np.ones(1), 2, np.ones(2)], [np.ones(1), (0,), np.ones(0)], [np.ones((1, 2)), (0, 2), np.ones((0, 2))], [np.ones((2, 1)), (2, 0), np.ones((2, 0))]]
    for input_array, shape, expected in data:
        actual = broadcast_to(input_array, shape)
        assert_array_equal(expected, actual)