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
def test_adjust_sigmoid_cutoff_half():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to half and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[1, 1, 2, 2, 3, 3, 4, 5], [5, 6, 7, 9, 10, 12, 14, 16], [19, 22, 25, 29, 34, 39, 44, 50], [57, 64, 72, 80, 89, 99, 108, 118], [128, 138, 148, 158, 167, 176, 184, 192], [199, 205, 211, 217, 221, 226, 229, 233], [236, 238, 240, 242, 244, 246, 247, 248], [249, 250, 250, 251, 251, 252, 252, 253]], dtype=np.uint8)
    result = exposure.adjust_sigmoid(image, 0.5, 10)
    assert_array_equal(result, expected)