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
def test_correlate17(self):
    array = numpy.array([1, 2, 3])
    tcor = [3, 5, 6]
    tcov = [2, 3, 5]
    kernel = numpy.array([1, 1])
    output = ndimage.correlate(array, kernel, origin=-1)
    assert_array_almost_equal(tcor, output)
    output = ndimage.convolve(array, kernel, origin=-1)
    assert_array_almost_equal(tcov, output)
    output = ndimage.correlate1d(array, kernel, origin=-1)
    assert_array_almost_equal(tcor, output)
    output = ndimage.convolve1d(array, kernel, origin=-1)
    assert_array_almost_equal(tcov, output)