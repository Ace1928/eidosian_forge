import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_hit_or_miss01(self, dtype):
    struct = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    expected = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    data = numpy.array([[0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype)
    out = numpy.zeros(data.shape, bool)
    ndimage.binary_hit_or_miss(data, struct, output=out)
    assert_array_almost_equal(expected, out)