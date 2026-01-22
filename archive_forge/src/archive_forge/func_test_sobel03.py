import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
@pytest.mark.parametrize('dtype', types + complex_types)
def test_sobel03(self, dtype):
    array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
    t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
    t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 0)
    output = numpy.zeros(array.shape, dtype)
    output = ndimage.sobel(array, 1)
    assert_array_almost_equal(t, output)