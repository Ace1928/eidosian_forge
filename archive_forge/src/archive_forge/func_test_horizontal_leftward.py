import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_horizontal_leftward():
    prof = profile_line(image, (0, 8), (0, 2), order=0, mode='constant')
    expected_prof = np.arange(8, 1, -1)
    assert_equal(prof, expected_prof)