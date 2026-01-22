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
def test_generic_filter1d01(self, dtype):
    weights = numpy.array([1.1, 2.2, 3.3])

    def _filter_func(input, output, fltr, total):
        fltr = fltr / total
        for ii in range(input.shape[0] - 2):
            output[ii] = input[ii] * fltr[0]
            output[ii] += input[ii + 1] * fltr[1]
            output[ii] += input[ii + 2] * fltr[2]
    a = numpy.arange(12, dtype=dtype)
    a.shape = (3, 4)
    r1 = ndimage.correlate1d(a, weights / weights.sum(), 0, origin=-1)
    r2 = ndimage.generic_filter1d(a, _filter_func, 3, axis=0, origin=-1, extra_arguments=(weights,), extra_keywords={'total': weights.sum()})
    assert_array_almost_equal(r1, r2)