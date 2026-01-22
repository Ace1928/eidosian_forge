import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_reduce_func_None_linewidth_3():
    prof = profile_line(pyth_image, (1, 2), (4, 2), linewidth=3, order=0, reduce_func=None, mode='constant')
    expected_prof = pyth_image[1:5, 1:4]
    assert_almost_equal(prof, expected_prof)