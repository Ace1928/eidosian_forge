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
def test_rescale_raises_on_incorrect_out_range():
    image = np.array([-128, 0, 127], dtype=np.int8)
    with pytest.raises(ValueError):
        _ = exposure.rescale_intensity(image, out_range='flat')