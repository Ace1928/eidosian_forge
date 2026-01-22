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
@pytest.mark.parametrize('filter_func, kwargs', [(ndimage.minimum_filter, {}), (ndimage.maximum_filter, {}), (ndimage.median_filter, {}), (ndimage.rank_filter, dict(rank=3)), (ndimage.percentile_filter, dict(percentile=30))])
@pytest.mark.parametrize('axes', [(0,), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize('separable_footprint', [False, True])
def test_filter_invalid_footprint_ndim(self, filter_func, kwargs, axes, separable_footprint):
    array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
    footprint = numpy.ones((3,) * (len(axes) + 1))
    if not separable_footprint:
        footprint[(0,) * footprint.ndim] = 0
    if filter_func in [ndimage.minimum_filter, ndimage.maximum_filter] and separable_footprint:
        match = 'sequence argument must have length equal to input rank'
    else:
        match = 'footprint array has incorrect shape'
    with pytest.raises(RuntimeError, match=match):
        filter_func(array, **kwargs, footprint=footprint, axes=axes)