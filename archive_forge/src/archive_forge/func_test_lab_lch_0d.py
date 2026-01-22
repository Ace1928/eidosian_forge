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
def test_lab_lch_0d(self):
    lab0 = self._get_lab0()
    lch0 = lab2lch(lab0)
    lch2 = lab2lch(lab0[None, None, :])
    assert_array_almost_equal(lch0, lch2[0, 0, :])