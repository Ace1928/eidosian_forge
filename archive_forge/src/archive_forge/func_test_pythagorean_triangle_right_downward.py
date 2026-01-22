import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_pythagorean_triangle_right_downward():
    prof = profile_line(image, (1, 1), (7, 9), order=0, mode='constant')
    expected_prof = np.array([11, 22, 23, 33, 34, 45, 56, 57, 67, 68, 79])
    assert_equal(prof, expected_prof)