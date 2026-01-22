import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', ['mean', 'median', 'minimum', 'maximum'])
def test_same_prepend_append(self, mode):
    """ Test that appended and prepended values are equal """
    a = np.array([-1, 2, -1]) + np.array([0, 1e-12, 0], dtype=np.float64)
    a = np.pad(a, (1, 1), mode)
    assert_equal(a[0], a[-1])