import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform13(self, order):
    data = numpy.ones([2], numpy.float64)

    def mapping(x):
        return (x[0] // 2,)
    out = ndimage.geometric_transform(data, mapping, [4], order=order)
    assert_array_almost_equal(out, [1, 1, 1, 1])