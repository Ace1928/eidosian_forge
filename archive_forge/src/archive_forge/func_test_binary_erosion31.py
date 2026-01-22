import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_binary_erosion31(self):
    struct = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    expected = [[0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1]]
    data = numpy.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]], bool)
    out = numpy.zeros(data.shape, bool)
    ndimage.binary_erosion(data, struct, border_value=1, iterations=1, output=out, origin=(-1, -1))
    assert_array_almost_equal(out, expected)