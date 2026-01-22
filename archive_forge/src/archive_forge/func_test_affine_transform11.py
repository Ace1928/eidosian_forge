import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform11(self, order):
    data = [1, 5, 2, 6, 3, 7, 4, 4]
    out = ndimage.affine_transform(data, [[2]], 0, (4,), order=order)
    assert_array_almost_equal(out, [1, 2, 3, 4])