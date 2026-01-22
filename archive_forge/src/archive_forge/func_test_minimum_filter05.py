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
def test_minimum_filter05(self):
    array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
    filter_shape = numpy.array([2, 3])
    output = ndimage.minimum_filter(array, filter_shape)
    assert_array_almost_equal([[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [5, 3, 3, 1, 1]], output)