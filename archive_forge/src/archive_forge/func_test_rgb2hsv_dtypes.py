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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_rgb2hsv_dtypes(dtype):
    rgb = img_as_float(data.colorwheel())[::16, ::16]
    rgb = rgb.astype(dtype=dtype, copy=False)
    hsv = rgb2hsv(rgb).reshape(-1, 3)
    float_dtype = _supported_float_type(dtype)
    assert hsv.dtype == float_dtype
    gt = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2]) for pt in rgb.reshape(-1, 3)])
    decimal = 3 if float_dtype == np.float32 else 7
    assert_array_almost_equal(hsv, gt, decimal=decimal)