import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_map_coordinates03(self):
    data = numpy.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]], order='F')
    idx = numpy.indices(data.shape) - 1
    out = ndimage.map_coordinates(data, idx)
    assert_array_almost_equal(out, [[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]])
    assert_array_almost_equal(out, ndimage.shift(data, (1, 1)))
    idx = numpy.indices(data[::2].shape) - 1
    out = ndimage.map_coordinates(data[::2], idx)
    assert_array_almost_equal(out, [[0, 0, 0, 0], [0, 4, 1, 3]])
    assert_array_almost_equal(out, ndimage.shift(data[::2], (1, 1)))
    idx = numpy.indices(data[:, ::2].shape) - 1
    out = ndimage.map_coordinates(data[:, ::2], idx)
    assert_array_almost_equal(out, [[0, 0], [0, 4], [0, 7]])
    assert_array_almost_equal(out, ndimage.shift(data[:, ::2], (1, 1)))