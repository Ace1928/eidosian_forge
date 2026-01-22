import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_output_combo(self):
    im = np.array([[0], [1], [2], [3]], dtype=np.uint8)
    result = graycomatrix(im, [1, 2], [0, np.pi / 2], 4)
    assert result.shape == (4, 4, 2, 2)
    z = np.zeros((4, 4), dtype=np.uint32)
    e1 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint32)
    e2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 0], z)
    np.testing.assert_array_equal(result[:, :, 1, 0], z)
    np.testing.assert_array_equal(result[:, :, 0, 1], e1)
    np.testing.assert_array_equal(result[:, :, 1, 1], e2)