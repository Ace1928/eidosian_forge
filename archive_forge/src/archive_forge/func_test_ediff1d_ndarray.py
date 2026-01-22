import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_ediff1d_ndarray(self):
    x = np.arange(5)
    test = ediff1d(x)
    control = array([1, 1, 1, 1], mask=[0, 0, 0, 0])
    assert_equal(test, control)
    assert_(isinstance(test, MaskedArray))
    assert_equal(test.filled(0), control.filled(0))
    assert_equal(test.mask, control.mask)
    test = ediff1d(x, to_end=masked, to_begin=masked)
    control = array([0, 1, 1, 1, 1, 0], mask=[1, 0, 0, 0, 0, 1])
    assert_(isinstance(test, MaskedArray))
    assert_equal(test.filled(0), control.filled(0))
    assert_equal(test.mask, control.mask)