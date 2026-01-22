import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_pythagorean_triangle_right_downward_interpolated():
    prof = profile_line(image, (1, 1), (7, 9), order=1, mode='constant')
    expected_prof = np.linspace(11, 79, 11)
    assert_almost_equal(prof, expected_prof)