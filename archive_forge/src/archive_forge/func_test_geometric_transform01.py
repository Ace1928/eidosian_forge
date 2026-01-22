import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform01(self, order):
    data = numpy.array([1])

    def mapping(x):
        return x
    out = ndimage.geometric_transform(data, mapping, data.shape, order=order)
    assert_array_almost_equal(out, [1])