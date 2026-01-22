import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
def test_rescale_named_in_range():
    image = np.array([0, uint10_max, uint10_max + 100], dtype=np.uint16)
    out = exposure.rescale_intensity(image, in_range='uint10')
    assert_array_almost_equal(out, [0, uint16_max, uint16_max])