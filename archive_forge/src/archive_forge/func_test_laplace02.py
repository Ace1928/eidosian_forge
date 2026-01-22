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
@pytest.mark.parametrize('dtype', [numpy.int32, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
def test_laplace02(self, dtype):
    array = numpy.array([[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], dtype) * 100
    tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
    tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
    output = numpy.zeros(array.shape, dtype)
    ndimage.laplace(array, output=output)
    assert_array_almost_equal(tmp1 + tmp2, output)