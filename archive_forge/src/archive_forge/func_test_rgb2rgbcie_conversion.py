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
def test_rgb2rgbcie_conversion(self, channel_axis):
    gt = np.array([[[0.1488856, 0.18288098, 0.19277574], [0.01163224, 0.16649536, 0.18948516], [0.12259182, 0.03308008, 0.17298223], [-0.01466154, 0.01669446, 0.16969164]], [[0.16354714, 0.16618652, 0.0230841], [0.02629378, 0.1498009, 0.01979351], [0.13725336, 0.01638562, 0.00329059], [0.0, 0.0, 0.0]]])
    img = np.moveaxis(self.colbars_array, source=-1, destination=channel_axis)
    out = rgb2rgbcie(img, channel_axis=channel_axis)
    out = np.moveaxis(out, source=channel_axis, destination=-1)
    assert_almost_equal(out, gt)