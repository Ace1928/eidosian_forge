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
def test_adapthist_borders():
    """Test border processing"""
    img = rgb2gray(util.img_as_float(data.astronaut()))
    img /= 100.0
    img[img.shape[0] // 2, img.shape[1] // 2] = 1.0
    border_index = -1
    for kernel_size in range(51, 71, 2):
        adapted = exposure.equalize_adapthist(img, kernel_size, clip_limit=0.5)
        assert norm_brightness_err(adapted[:, border_index], img[:, border_index]) > 0.1
        assert norm_brightness_err(adapted[border_index, :], img[border_index, :]) > 0.1