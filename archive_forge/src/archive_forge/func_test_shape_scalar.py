import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_shape_scalar(self):
    b = atleast_1d(1.0)
    assert_equal(b.shape, (1,))
    assert_equal(b.mask.shape, b.shape)
    assert_equal(b.data.shape, b.shape)
    b = atleast_1d(1.0, 2.0)
    for a in b:
        assert_equal(a.shape, (1,))
        assert_equal(a.mask.shape, a.shape)
        assert_equal(a.data.shape, a.shape)
    b = atleast_2d(1.0)
    assert_equal(b.shape, (1, 1))
    assert_equal(b.mask.shape, b.shape)
    assert_equal(b.data.shape, b.shape)
    b = atleast_2d(1.0, 2.0)
    for a in b:
        assert_equal(a.shape, (1, 1))
        assert_equal(a.mask.shape, a.shape)
        assert_equal(a.data.shape, a.shape)
    b = atleast_3d(1.0)
    assert_equal(b.shape, (1, 1, 1))
    assert_equal(b.mask.shape, b.shape)
    assert_equal(b.data.shape, b.shape)
    b = atleast_3d(1.0, 2.0)
    for a in b:
        assert_equal(a.shape, (1, 1, 1))
        assert_equal(a.mask.shape, a.shape)
        assert_equal(a.data.shape, a.shape)
    b = diagflat(1.0)
    assert_equal(b.shape, (1, 1))
    assert_equal(b.mask.shape, b.data.shape)