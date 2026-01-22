import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(2, 6))
@pytest.mark.parametrize('dtype', types)
def test_spline02(self, dtype, order):
    data = numpy.array([1], dtype)
    out = ndimage.spline_filter(data, order=order)
    assert_array_almost_equal(out, [1])