import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_corner_orientations_square(dtype):
    square = np.zeros((12, 12), dtype=dtype)
    square[3:9, 3:9] = 1
    corners = corner_peaks(corner_fast(square, 9), min_distance=1, threshold_rel=0)
    actual_orientations = corner_orientations(square, corners, octagon(3, 2))
    assert actual_orientations.dtype == _supported_float_type(dtype)
    actual_orientations_degrees = np.rad2deg(actual_orientations)
    expected_orientations_degree = np.array([45, 135, -45, -135])
    assert_array_equal(actual_orientations_degrees, expected_orientations_degree)