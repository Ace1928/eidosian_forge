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
def test_rescale_float_output():
    image = np.array([-128, 0, 127], dtype=np.int8)
    output_image = exposure.rescale_intensity(image, out_range=(0, 255))
    assert_array_equal(output_image, [0, 128, 255])
    assert output_image.dtype == float