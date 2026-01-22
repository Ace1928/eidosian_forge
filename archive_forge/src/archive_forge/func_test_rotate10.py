import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_rotate10(self):
    data = numpy.arange(45, dtype=numpy.float64).reshape((3, 5, 3))
    expected = numpy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.54914793, 7.54914793, 8.54914793], [10.84520162, 11.84520162, 12.84520162], [0.0, 0.0, 0.0]], [[6.19286575, 7.19286575, 8.19286575], [13.4730712, 14.4730712, 15.4730712], [21.0, 22.0, 23.0], [28.5269288, 29.5269288, 30.5269288], [35.80713425, 36.80713425, 37.80713425]], [[0.0, 0.0, 0.0], [31.15479838, 32.15479838, 33.15479838], [35.45085207, 36.45085207, 37.45085207], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    out = ndimage.rotate(data, angle=12, reshape=False)
    assert_array_almost_equal(out, expected)