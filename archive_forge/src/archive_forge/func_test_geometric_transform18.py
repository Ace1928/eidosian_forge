import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform18(self, order):
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    def mapping(x):
        return (x[0] * 2, x[1] * 2)
    out = ndimage.geometric_transform(data, mapping, (1, 2), order=order)
    assert_array_almost_equal(out, [[1, 3]])