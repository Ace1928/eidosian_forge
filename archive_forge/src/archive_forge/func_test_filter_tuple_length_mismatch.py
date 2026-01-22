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
@pytest.mark.parametrize('n_mismatch', [1, 3])
@pytest.mark.parametrize('filter_func, kwargs, key, val', _cases_axes_tuple_length_mismatch())
def test_filter_tuple_length_mismatch(self, n_mismatch, filter_func, kwargs, key, val):
    array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
    kwargs = dict(**kwargs, axes=(0, 1))
    kwargs[key] = (val,) * n_mismatch
    err_msg = 'sequence argument must have length equal to input rank'
    with pytest.raises(RuntimeError, match=err_msg):
        filter_func(array, **kwargs)