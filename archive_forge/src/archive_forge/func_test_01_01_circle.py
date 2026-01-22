import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_01_01_circle(self):
    """Test that the Canny filter finds the outlines of a circle"""
    i, j = np.mgrid[-200:200, -200:200].astype(float) / 200
    c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
    result = feature.canny(c.astype(float), 4, 0, 0, np.ones(c.shape, bool))
    cd = binary_dilation(c, iterations=3)
    ce = binary_erosion(c, iterations=3)
    cde = np.logical_and(cd, np.logical_not(ce))
    self.assertTrue(np.all(cde[result]))
    point_count = np.sum(result)
    self.assertTrue(point_count > 1200)
    self.assertTrue(point_count < 1600)