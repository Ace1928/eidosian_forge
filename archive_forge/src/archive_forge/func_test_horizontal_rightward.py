import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_horizontal_rightward():
    prof = profile_line(image, (0, 2), (0, 8), order=0, mode='constant')
    expected_prof = np.arange(2, 9)
    assert_equal(prof, expected_prof)