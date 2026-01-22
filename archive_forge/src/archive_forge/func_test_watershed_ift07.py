import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_watershed_ift07(self):
    shape = (7, 6)
    data = np.zeros(shape, dtype=np.uint8)
    data = data.transpose()
    data[...] = np.array([[0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.uint8)
    markers = np.array([[-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], np.int8)
    out = np.zeros(shape, dtype=np.int16)
    out = out.transpose()
    ndimage.watershed_ift(data, markers, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], output=out)
    expected = [[-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
    assert_array_almost_equal(out, expected)