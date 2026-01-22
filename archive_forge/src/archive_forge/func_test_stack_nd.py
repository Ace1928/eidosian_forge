import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_stack_nd(self):
    shp = (3, 2)
    d1 = np.random.randint(0, 10, shp)
    d2 = np.random.randint(0, 10, shp)
    m1 = np.random.randint(0, 2, shp).astype(bool)
    m2 = np.random.randint(0, 2, shp).astype(bool)
    a1 = masked_array(d1, mask=m1)
    a2 = masked_array(d2, mask=m2)
    c = stack([a1, a2], axis=0)
    c_shp = (2,) + shp
    assert_equal(c.shape, c_shp)
    assert_array_equal(a1.mask, c[0].mask)
    assert_array_equal(a2.mask, c[1].mask)
    c = stack([a1, a2], axis=-1)
    c_shp = shp + (2,)
    assert_equal(c.shape, c_shp)
    assert_array_equal(a1.mask, c[..., 0].mask)
    assert_array_equal(a2.mask, c[..., 1].mask)
    shp = (3, 2, 4, 5)
    d1 = np.random.randint(0, 10, shp)
    d2 = np.random.randint(0, 10, shp)
    m1 = np.random.randint(0, 2, shp).astype(bool)
    m2 = np.random.randint(0, 2, shp).astype(bool)
    a1 = masked_array(d1, mask=m1)
    a2 = masked_array(d2, mask=m2)
    c = stack([a1, a2], axis=0)
    c_shp = (2,) + shp
    assert_equal(c.shape, c_shp)
    assert_array_equal(a1.mask, c[0].mask)
    assert_array_equal(a2.mask, c[1].mask)
    c = stack([a1, a2], axis=-1)
    c_shp = shp + (2,)
    assert_equal(c.shape, c_shp)
    assert_array_equal(a1.mask, c[..., 0].mask)
    assert_array_equal(a2.mask, c[..., 1].mask)