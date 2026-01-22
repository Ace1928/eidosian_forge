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
def test_hed_rgb_roundtrip(self):
    img_in = img_as_ubyte(self.img_stains)
    img_out = rgb2hed(hed2rgb(img_in))
    assert_equal(img_as_ubyte(img_out), img_in)