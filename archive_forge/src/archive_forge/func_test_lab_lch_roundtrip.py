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
def test_lab_lch_roundtrip(self, channel_axis):
    rgb = img_as_float(self.img_rgb)
    rgb = np.moveaxis(rgb, source=-1, destination=channel_axis)
    lab = rgb2lab(rgb, channel_axis=channel_axis)
    lab2 = lch2lab(lab2lch(lab, channel_axis=channel_axis), channel_axis=channel_axis)
    assert_array_almost_equal(lab2, lab)