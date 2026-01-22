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
def test_rank04(self):
    array = numpy.array([3, 2, 5, 1, 4])
    expected = [3, 3, 2, 4, 4]
    output = ndimage.rank_filter(array, 1, size=3)
    assert_array_almost_equal(expected, output)
    output = ndimage.percentile_filter(array, 50, size=3)
    assert_array_almost_equal(expected, output)
    output = ndimage.median_filter(array, size=3)
    assert_array_almost_equal(expected, output)