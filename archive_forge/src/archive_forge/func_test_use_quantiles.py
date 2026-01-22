import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_use_quantiles(self):
    image = img_as_float(data.camera()[::100, ::100])
    correct_output = np.array([[False, False, False, False, False, False], [False, True, True, True, False, False], [False, False, False, True, False, False], [False, False, False, True, False, False], [False, False, True, True, False, False], [False, False, False, False, False, False]])
    result = feature.canny(image, low_threshold=0.6, high_threshold=0.8, use_quantiles=True)
    assert_equal(result, correct_output)