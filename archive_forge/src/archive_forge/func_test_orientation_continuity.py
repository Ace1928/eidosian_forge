import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_orientation_continuity():
    arr1 = np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    arr2 = np.array([[0, 0, 0, 2], [0, 0, 2, 0], [0, 2, 0, 0], [2, 0, 0, 0]])
    arr3 = np.array([[0, 0, 0, 3], [0, 0, 3, 3], [0, 3, 0, 0], [3, 0, 0, 0]])
    image = np.hstack((arr1, arr2, arr3))
    props = regionprops(image)
    orientations = [prop.orientation for prop in props]
    np.testing.assert_allclose(orientations, orientations[1], rtol=0, atol=0.08)
    assert_almost_equal(orientations[0], -0.7144496360953664)
    assert_almost_equal(orientations[1], -0.7853981633974483)
    assert_almost_equal(orientations[2], -0.8563466906995303)
    spacing = (3.2, 1.2)
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].moments_weighted_central
    centralMpq = get_central_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    assert_almost_equal(wmu[0, 0], centralMpq(0, 0))
    assert_almost_equal(wmu[0, 1], centralMpq(0, 1))
    assert_almost_equal(wmu[0, 2], centralMpq(0, 2))
    assert_almost_equal(wmu[0, 3], centralMpq(0, 3))
    assert_almost_equal(wmu[1, 0], centralMpq(1, 0))
    assert_almost_equal(wmu[1, 1], centralMpq(1, 1))
    assert_almost_equal(wmu[1, 2], centralMpq(1, 2))
    assert_almost_equal(wmu[1, 3], centralMpq(1, 3))
    assert_almost_equal(wmu[2, 0], centralMpq(2, 0))
    assert_almost_equal(wmu[2, 1], centralMpq(2, 1))
    assert_almost_equal(wmu[2, 2], centralMpq(2, 2))
    assert_almost_equal(wmu[2, 3], centralMpq(2, 3))
    assert_almost_equal(wmu[3, 0], centralMpq(3, 0))
    assert_almost_equal(wmu[3, 1], centralMpq(3, 1))
    assert_almost_equal(wmu[3, 2], centralMpq(3, 2))
    assert_almost_equal(wmu[3, 3], centralMpq(3, 3))