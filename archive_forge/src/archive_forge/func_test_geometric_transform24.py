import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_geometric_transform24(self, order):
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    def mapping(x, a, b):
        return (a, x[0] * b)
    out = ndimage.geometric_transform(data, mapping, (2,), order=order, extra_arguments=(1,), extra_keywords={'b': 2})
    assert_array_almost_equal(out, [5, 7])