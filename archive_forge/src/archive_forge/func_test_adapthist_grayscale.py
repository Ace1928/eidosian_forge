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
def test_adapthist_grayscale(dtype):
    """Test a grayscale float image"""
    img = util.img_as_float(data.astronaut()).astype(dtype, copy=False)
    img = rgb2gray(img)
    img = np.dstack((img, img, img))
    adapted = exposure.equalize_adapthist(img, kernel_size=(57, 51), clip_limit=0.01, nbins=128)
    assert img.shape == adapted.shape
    assert adapted.dtype == _supported_float_type(dtype)
    snr_decimal = 3 if dtype != np.float16 else 2
    assert_almost_equal(peak_snr(img, adapted), 100.14, snr_decimal)
    assert_almost_equal(norm_brightness_err(img, adapted), 0.0529, 3)