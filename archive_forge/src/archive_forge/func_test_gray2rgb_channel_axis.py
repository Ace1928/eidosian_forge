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
@pytest.mark.parametrize('shape', [(5, 5), (5, 5, 4), (5, 4, 5, 4)])
@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
def test_gray2rgb_channel_axis(shape, channel_axis):
    img = np.random.random(shape)
    rgb = gray2rgb(img, channel_axis=channel_axis)
    assert rgb.ndim == img.ndim + 1
    new_axis_loc = channel_axis % rgb.ndim
    assert_equal(rgb.shape, shape[:new_axis_loc] + (3,) + shape[new_axis_loc:])
    assert rgb.dtype == img.dtype