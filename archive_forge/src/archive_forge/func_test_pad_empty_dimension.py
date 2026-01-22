import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_pad_empty_dimension(self):
    arr = np.zeros((3, 0, 2))
    result = np.pad(arr, [(0,), (2,), (1,)], mode='empty')
    assert result.shape == (3, 4, 4)