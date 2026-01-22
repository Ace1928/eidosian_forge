import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_distance_transform_edt01(self, dtype):
    data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
    out, ft = ndimage.distance_transform_edt(data, return_indices=True)
    bf = ndimage.distance_transform_bf(data, 'euclidean')
    assert_array_almost_equal(bf, out)
    dt = ft - numpy.indices(ft.shape[1:], dtype=ft.dtype)
    dt = dt.astype(numpy.float64)
    numpy.multiply(dt, dt, dt)
    dt = numpy.add.reduce(dt, axis=0)
    numpy.sqrt(dt, dt)
    assert_array_almost_equal(bf, dt)