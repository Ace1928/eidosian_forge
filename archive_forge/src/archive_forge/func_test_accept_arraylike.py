import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
def test_accept_arraylike(self):
    actual = geometric_slerp([1, 0], [0, 1], [0, 1 / 3, 0.5, 2 / 3, 1])
    expected = np.array([[1, 0], [np.sqrt(3) / 2, 0.5], [np.sqrt(2) / 2, np.sqrt(2) / 2], [0.5, np.sqrt(3) / 2], [0, 1]], dtype=np.float64)
    assert_allclose(actual, expected, atol=1e-16)