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
def test_multiple_modes_sequentially():
    arr = numpy.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    modes = ['reflect', 'wrap']
    expected = ndimage.gaussian_filter1d(arr, 1, axis=0, mode=modes[0])
    expected = ndimage.gaussian_filter1d(expected, 1, axis=1, mode=modes[1])
    assert_equal(expected, ndimage.gaussian_filter(arr, 1, mode=modes))
    expected = ndimage.uniform_filter1d(arr, 5, axis=0, mode=modes[0])
    expected = ndimage.uniform_filter1d(expected, 5, axis=1, mode=modes[1])
    assert_equal(expected, ndimage.uniform_filter(arr, 5, mode=modes))
    expected = ndimage.maximum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = ndimage.maximum_filter1d(expected, size=5, axis=1, mode=modes[1])
    assert_equal(expected, ndimage.maximum_filter(arr, size=5, mode=modes))
    expected = ndimage.minimum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = ndimage.minimum_filter1d(expected, size=5, axis=1, mode=modes[1])
    assert_equal(expected, ndimage.minimum_filter(arr, size=5, mode=modes))