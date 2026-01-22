import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_vertical_upward():
    prof = profile_line(image, (8, 5), (2, 5), order=0, mode='constant')
    expected_prof = np.arange(85, 15, -10)
    assert_equal(prof, expected_prof)