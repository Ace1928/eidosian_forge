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
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_bro_rgb_roundtrip_float(self, channel_axis):
    from skimage.color.colorconv import bro_from_rgb, rgb_from_bro
    img_in = self.img_stains
    img_in = np.moveaxis(img_in, source=-1, destination=channel_axis)
    img_out = combine_stains(img_in, rgb_from_bro, channel_axis=channel_axis)
    img_out = separate_stains(img_out, bro_from_rgb, channel_axis=channel_axis)
    assert_array_almost_equal(img_out, img_in)