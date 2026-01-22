import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_ror(self):
    lbp = local_binary_pattern(self.image, 8, 1, 'ror')
    ref = np.array([[0, 127, 0, 255, 3, 255], [31, 0, 5, 51, 1, 7], [119, 255, 3, 127, 0, 63], [3, 1, 31, 63, 31, 0], [255, 1, 255, 95, 0, 127], [3, 5, 0, 255, 1, 3]])
    np.testing.assert_array_equal(lbp, ref)