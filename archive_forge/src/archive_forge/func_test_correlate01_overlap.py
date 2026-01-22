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
def test_correlate01_overlap(self):
    array = numpy.arange(256).reshape(16, 16)
    weights = numpy.array([2])
    expected = 2 * array
    ndimage.correlate1d(array, weights, output=array)
    assert_array_almost_equal(array, expected)