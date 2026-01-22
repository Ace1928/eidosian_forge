import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
@pytest.mark.parametrize('mode', _all_modes.keys())
def test_misshaped_pad_width_2(self, mode):
    arr = np.arange(30).reshape((6, 5))
    match = 'input operand has more dimensions than allowed by the axis remapping'
    with pytest.raises(ValueError, match=match):
        np.pad(arr, (((3,), (4,), (5,)), ((0,), (1,), (2,))), mode)