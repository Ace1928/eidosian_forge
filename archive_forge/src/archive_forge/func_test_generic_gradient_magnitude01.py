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
def test_generic_gradient_magnitude01(self):
    array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], numpy.float64)

    def derivative(input, axis, output, mode, cval, a, b):
        sigma = [a, b / 2.0]
        input = numpy.asarray(input)
        order = [0] * input.ndim
        order[axis] = 1
        return ndimage.gaussian_filter(input, sigma, order, output, mode, cval)
    tmp1 = ndimage.gaussian_gradient_magnitude(array, 1.0)
    tmp2 = ndimage.generic_gradient_magnitude(array, derivative, extra_arguments=(1.0,), extra_keywords={'b': 2.0})
    assert_array_almost_equal(tmp1, tmp2)