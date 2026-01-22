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
def test_gray2rgb():
    x = np.array([0, 0.5, 1])
    w = gray2rgb(x)
    expected_output = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    assert_equal(w, expected_output)
    x = x.reshape((3, 1))
    y = gray2rgb(x)
    assert_equal(y.shape, (3, 1, 3))
    assert_equal(y.dtype, x.dtype)
    assert_equal(y[..., 0], x)
    assert_equal(y[0, 0, :], [0, 0, 0])
    x = np.array([[0, 128, 255]], dtype=np.uint8)
    z = gray2rgb(x)
    assert_equal(z.shape, (1, 3, 3))
    assert_equal(z[..., 0], x)
    assert_equal(z[0, 1, :], [128, 128, 128])