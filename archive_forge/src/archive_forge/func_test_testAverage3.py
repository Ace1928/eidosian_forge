import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_testAverage3(self):
    a = arange(6)
    b = arange(6) * 3
    r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
    assert_equal(shape(r1), shape(w1))
    assert_equal(r1.shape, w1.shape)
    r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
    assert_equal(shape(w2), shape(r2))
    r2, w2 = average(ones((2, 2, 3)), returned=True)
    assert_equal(shape(w2), shape(r2))
    r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
    assert_equal(shape(w2), shape(r2))
    a2d = array([[1, 2], [0, 4]], float)
    a2dm = masked_array(a2d, [[False, False], [True, False]])
    a2da = average(a2d, axis=0)
    assert_equal(a2da, [0.5, 3.0])
    a2dma = average(a2dm, axis=0)
    assert_equal(a2dma, [1.0, 3.0])
    a2dma = average(a2dm, axis=None)
    assert_equal(a2dma, 7.0 / 3.0)
    a2dma = average(a2dm, axis=1)
    assert_equal(a2dma, [1.5, 4.0])