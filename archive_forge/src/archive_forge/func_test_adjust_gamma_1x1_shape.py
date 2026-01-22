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
def test_adjust_gamma_1x1_shape():
    """Check that the shape is maintained"""
    img = np.ones([1, 1])
    result = exposure.adjust_gamma(img, 1.5)
    assert img.shape == result.shape