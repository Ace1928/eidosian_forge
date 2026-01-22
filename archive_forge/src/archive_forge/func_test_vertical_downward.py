import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_vertical_downward():
    prof = profile_line(image, (2, 5), (8, 5), order=0, mode='constant')
    expected_prof = np.arange(25, 95, 10)
    assert_equal(prof, expected_prof)