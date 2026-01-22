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
def test_correlate12(self):
    array = numpy.array([[1, 2, 3], [4, 5, 6]])
    kernel = numpy.array([[1, 0], [0, 1]])
    output = ndimage.correlate(array, kernel)
    assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
    output = ndimage.convolve(array, kernel)
    assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)