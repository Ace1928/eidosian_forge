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
def test_moments_weighted_spacing():
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted
    ref = np.array([[74.0, 699.0, 7863.0, 97317.0], [410.0, 3785.0, 44063.0, 572567.0], [2750.0, 24855.0, 293477.0, 3900717.0], [19778.0, 175001.0, 2081051.0, 28078871.0]])
    assert_array_almost_equal(wm, ref)
    spacing = (3.2, 1.2)
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].moments_weighted
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    assert_almost_equal(wmu[0, 0], Mpq(0, 0))
    assert_almost_equal(wmu[0, 1], Mpq(0, 1))
    assert_almost_equal(wmu[0, 2], Mpq(0, 2))
    assert_almost_equal(wmu[0, 3], Mpq(0, 3))
    assert_almost_equal(wmu[1, 0], Mpq(1, 0))
    assert_almost_equal(wmu[1, 1], Mpq(1, 1))
    assert_almost_equal(wmu[1, 2], Mpq(1, 2))
    assert_almost_equal(wmu[1, 3], Mpq(1, 3))
    assert_almost_equal(wmu[2, 0], Mpq(2, 0))
    assert_almost_equal(wmu[2, 1], Mpq(2, 1))
    assert_almost_equal(wmu[2, 2], Mpq(2, 2))
    assert_almost_equal(wmu[2, 3], Mpq(2, 3))
    assert_almost_equal(wmu[3, 0], Mpq(3, 0))
    assert_almost_equal(wmu[3, 1], Mpq(3, 1))
    assert_almost_equal(wmu[3, 2], Mpq(3, 2))
    assert_almost_equal(wmu[3, 3], Mpq(3, 3), decimal=6)