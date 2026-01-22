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
@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
def test_hsv2rgb_conversion(self, channel_axis):
    rgb = self.img_rgb.astype('float32')[::16, ::16]
    hsv = np.array([colorsys.rgb_to_hsv(pt[0], pt[1], pt[2]) for pt in rgb.reshape(-1, 3)]).reshape(rgb.shape)
    hsv = np.moveaxis(hsv, source=-1, destination=channel_axis)
    _rgb = hsv2rgb(hsv, channel_axis=channel_axis)
    _rgb = np.moveaxis(_rgb, source=channel_axis, destination=-1)
    assert_almost_equal(rgb, _rgb, decimal=4)