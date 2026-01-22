import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
def test_yuv(self):
    rgb = np.array([[[1.0, 1.0, 1.0]]])
    assert_array_almost_equal(rgb2yuv(rgb), np.array([[[1, 0, 0]]]))
    assert_array_almost_equal(rgb2yiq(rgb), np.array([[[1, 0, 0]]]))
    assert_array_almost_equal(rgb2ypbpr(rgb), np.array([[[1, 0, 0]]]))
    assert_array_almost_equal(rgb2ycbcr(rgb), np.array([[[235, 128, 128]]]))
    assert_array_almost_equal(rgb2ydbdr(rgb), np.array([[[1, 0, 0]]]))
    rgb = np.array([[[0.0, 1.0, 0.0]]])
    assert_array_almost_equal(rgb2yuv(rgb), np.array([[[0.587, -0.28886916, -0.51496512]]]))
    assert_array_almost_equal(rgb2yiq(rgb), np.array([[[0.587, -0.27455667, -0.52273617]]]))
    assert_array_almost_equal(rgb2ypbpr(rgb), np.array([[[0.587, -0.331264, -0.418688]]]))
    assert_array_almost_equal(rgb2ycbcr(rgb), np.array([[[144.553, 53.797, 34.214]]]))
    assert_array_almost_equal(rgb2ydbdr(rgb), np.array([[[0.587, -0.883, 1.116]]]))