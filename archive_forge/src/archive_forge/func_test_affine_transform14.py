import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform14(self, order):
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    out = ndimage.affine_transform(data, [[2, 0], [0, 1]], 0, (1, 4), order=order)
    assert_array_almost_equal(out, [[1, 2, 3, 4]])