import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('shift', [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
@pytest.mark.parametrize('order', range(0, 6))
def test_shift_grid_constant1(self, shift, order):
    x = numpy.arange(20).reshape((5, 4))
    assert_array_almost_equal(ndimage.shift(x, shift, mode='grid-constant', order=order), ndimage.shift(x, shift, mode='constant', order=order))