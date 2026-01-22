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
def test_adjust_gamma_negative():
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    with pytest.raises(ValueError):
        exposure.adjust_gamma(image, -1)