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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_adjust_log_1x1_shape(dtype):
    """Check that the shape is maintained"""
    img = np.ones([1, 1], dtype=dtype)
    result = exposure.adjust_log(img, 1)
    assert img.shape == result.shape
    assert result.dtype == dtype