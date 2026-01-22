import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
@pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
def test_zoom_affine01(self, order, dtype):
    data = numpy.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype)
    if data.dtype.kind == 'c':
        data -= 1j * data
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'The behavior of affine_transform with a 1-D array .* has changed')
        out = ndimage.affine_transform(data, [0.5, 0.5], 0, (6, 8), order=order)
    assert_array_almost_equal(out[::2, ::2], data)