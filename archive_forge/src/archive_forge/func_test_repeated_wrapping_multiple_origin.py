import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_repeated_wrapping_multiple_origin(self):
    """
        Assert that 'wrap' pads only with multiples of the original area if
        the pad width is larger than the original array.
        """
    a = np.arange(4).reshape(2, 2)
    a = np.pad(a, [(1, 3), (3, 1)], mode='wrap')
    b = np.array([[3, 2, 3, 2, 3, 2], [1, 0, 1, 0, 1, 0], [3, 2, 3, 2, 3, 2], [1, 0, 1, 0, 1, 0], [3, 2, 3, 2, 3, 2], [1, 0, 1, 0, 1, 0]])
    assert_array_equal(a, b)