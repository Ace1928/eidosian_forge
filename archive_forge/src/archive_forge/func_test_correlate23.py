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
@pytest.mark.parametrize('dtype_array', types)
@pytest.mark.parametrize('dtype_output', types)
def test_correlate23(self, dtype_array, dtype_output):
    weights = numpy.array([1, 2, 1])
    expected = [[5, 10, 15], [7, 14, 21]]
    array = numpy.array([[1, 2, 3], [2, 4, 6]], dtype_array)
    output = numpy.zeros((2, 3), dtype_output)
    ndimage.correlate1d(array, weights, axis=0, mode='nearest', output=output)
    assert_array_almost_equal(output, expected)
    ndimage.convolve1d(array, weights, axis=0, mode='nearest', output=output)
    assert_array_almost_equal(output, expected)