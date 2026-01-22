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
def test_adapthist_incorrect_kernel_size():
    img = np.ones((8, 8), dtype=float)
    with pytest.raises(ValueError, match='Incorrect value of `kernel_size`'):
        exposure.equalize_adapthist(img, (3, 3, 3))