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
def test_corner_peaks():
    response = np.zeros((10, 10))
    response[2:5, 2:5] = 1
    response[8:10, 0:2] = 1
    corners = corner_peaks(response, exclude_border=False, min_distance=10, threshold_rel=0)
    assert corners.shape == (1, 2)
    corners = corner_peaks(response, exclude_border=False, min_distance=5, threshold_rel=0)
    assert corners.shape == (2, 2)
    corners = corner_peaks(response, exclude_border=False, min_distance=1)
    assert corners.shape == (5, 2)
    corners = corner_peaks(response, exclude_border=False, min_distance=1, indices=False)
    assert np.sum(corners) == 5