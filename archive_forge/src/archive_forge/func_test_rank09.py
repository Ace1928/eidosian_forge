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
@pytest.mark.parametrize('dtype', types)
def test_rank09(self, dtype):
    expected = [[3, 3, 2, 4, 4], [3, 5, 2, 5, 1], [5, 5, 8, 3, 5]]
    footprint = [[1, 0, 1], [0, 1, 0]]
    array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype)
    output = ndimage.rank_filter(array, 1, footprint=footprint)
    assert_array_almost_equal(expected, output)
    output = ndimage.percentile_filter(array, 35, footprint=footprint)
    assert_array_almost_equal(expected, output)