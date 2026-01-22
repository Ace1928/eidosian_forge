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
def test_moments_weighted_central():
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted_central
    ref = np.array([[74.0, 3.7303493627e-14, 1260.2837838, -765.61796932], [-2.1316282073e-13, -87.837837838, 2157.1526662, -4238.5971907], [478.37837838, -148.01314828, 6698.979942, -9950.1164076], [-759.43608473, -1271.4707125, 15304.076361, -33156.729271]])
    assert_array_almost_equal(wmu, ref)
    centralMpq = get_central_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    assert_almost_equal(centralMpq(0, 0), ref[0, 0])
    assert_almost_equal(centralMpq(0, 1), ref[0, 1])
    assert_almost_equal(centralMpq(0, 2), ref[0, 2])
    assert_almost_equal(centralMpq(0, 3), ref[0, 3])
    assert_almost_equal(centralMpq(1, 0), ref[1, 0])
    assert_almost_equal(centralMpq(1, 1), ref[1, 1])
    assert_almost_equal(centralMpq(1, 2), ref[1, 2])
    assert_almost_equal(centralMpq(1, 3), ref[1, 3])
    assert_almost_equal(centralMpq(2, 0), ref[2, 0])
    assert_almost_equal(centralMpq(2, 1), ref[2, 1])
    assert_almost_equal(centralMpq(2, 2), ref[2, 2])
    assert_almost_equal(centralMpq(2, 3), ref[2, 3])
    assert_almost_equal(centralMpq(3, 0), ref[3, 0])
    assert_almost_equal(centralMpq(3, 1), ref[3, 1])
    assert_almost_equal(centralMpq(3, 2), ref[3, 2])
    assert_almost_equal(centralMpq(3, 3), ref[3, 3])