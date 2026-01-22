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
def test_uniform04(self):
    array = numpy.array([2, 4, 6])
    filter_shape = [2]
    output = ndimage.uniform_filter(array, filter_shape)
    assert_array_almost_equal([2, 3, 5], output)