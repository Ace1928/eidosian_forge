import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_mask_none(self):
    result1 = feature.canny(np.zeros((20, 20)), 4, 0, 0, np.ones((20, 20), bool))
    result2 = feature.canny(np.zeros((20, 20)), 4, 0, 0)
    self.assertTrue(np.all(result1 == result2))