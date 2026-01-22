import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_watershed_ift04(self):
    data = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
    markers = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], np.int8)
    out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    expected = [[-1, -1, -1, -1, -1, -1, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, 2, 2, 3, 3, 3, -1], [-1, -1, -1, -1, -1, -1, -1]]
    assert_array_almost_equal(out, expected)