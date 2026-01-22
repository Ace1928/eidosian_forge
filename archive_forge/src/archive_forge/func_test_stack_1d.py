import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_stack_1d(self):
    a = masked_array([0, 1, 2], mask=[0, 1, 0])
    b = masked_array([9, 8, 7], mask=[1, 0, 0])
    c = stack([a, b], axis=0)
    assert_equal(c.shape, (2, 3))
    assert_array_equal(a.mask, c[0].mask)
    assert_array_equal(b.mask, c[1].mask)
    d = vstack([a, b])
    assert_array_equal(c.data, d.data)
    assert_array_equal(c.mask, d.mask)
    c = stack([a, b], axis=1)
    assert_equal(c.shape, (3, 2))
    assert_array_equal(a.mask, c[:, 0].mask)
    assert_array_equal(b.mask, c[:, 1].mask)