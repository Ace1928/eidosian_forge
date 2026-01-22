import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', ['minimum', 'maximum'])
def test_zero_stat_length_invalid(self, mode):
    match = 'stat_length of 0 yields no value for padding'
    with pytest.raises(ValueError, match=match):
        np.pad([1.0, 2.0], 0, mode, stat_length=0)
    with pytest.raises(ValueError, match=match):
        np.pad([1.0, 2.0], 0, mode, stat_length=(1, 0))
    with pytest.raises(ValueError, match=match):
        np.pad([1.0, 2.0], 1, mode, stat_length=0)
    with pytest.raises(ValueError, match=match):
        np.pad([1.0, 2.0], 1, mode, stat_length=(1, 0))