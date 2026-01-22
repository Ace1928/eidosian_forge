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
@pytest.mark.parametrize('axes', tuple(itertools.combinations(range(-3, 3), 2)))
@pytest.mark.parametrize('filter_func, kwargs', [(ndimage.minimum_filter, {}), (ndimage.maximum_filter, {}), (ndimage.median_filter, {}), (ndimage.rank_filter, dict(rank=3)), (ndimage.percentile_filter, dict(percentile=60))])
def test_minmax_nonseparable_axes(self, filter_func, axes, kwargs):
    array = numpy.arange(6 * 8 * 12, dtype=numpy.float32).reshape(6, 8, 12)
    footprint = numpy.tri(5)
    axes = numpy.array(axes)
    if len(set(axes % array.ndim)) != len(axes):
        with pytest.raises(ValueError):
            filter_func(array, footprint=footprint, axes=axes, **kwargs)
        return
    output = filter_func(array, footprint=footprint, axes=axes, **kwargs)
    missing_axis = tuple(set(range(3)) - set(axes % array.ndim))[0]
    footprint_3d = numpy.expand_dims(footprint, missing_axis)
    expected = filter_func(array, footprint=footprint_3d, **kwargs)
    assert_allclose(output, expected)