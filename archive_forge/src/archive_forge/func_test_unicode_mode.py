import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_unicode_mode():
    a = np.pad([1], 2, mode='constant')
    b = np.array([0, 0, 1, 0, 0])
    assert_array_equal(a, b)