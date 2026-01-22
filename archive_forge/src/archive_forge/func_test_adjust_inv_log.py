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
def test_adjust_inv_log():
    """Verifying the output with expected results for inverse logarithmic
    correction with multiplier constant multiplier equal to unity"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[0, 2, 5, 8, 11, 14, 17, 20], [23, 26, 29, 32, 35, 38, 41, 45], [48, 51, 55, 58, 61, 65, 68, 72], [76, 79, 83, 87, 90, 94, 98, 102], [106, 110, 114, 118, 122, 126, 130, 134], [138, 143, 147, 151, 156, 160, 165, 170], [174, 179, 184, 188, 193, 198, 203, 208], [213, 218, 224, 229, 234, 239, 245, 250]], dtype=np.uint8)
    result = exposure.adjust_log(image, 1, True)
    assert_array_equal(result, expected)