import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_non_masked(self):
    x = np.arange(9)
    assert_equal(np.ma.median(x), 4.0)
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = range(8)
    assert_equal(np.ma.median(x), 3.5)
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = 5
    assert_equal(np.ma.median(x), 5.0)
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = np.arange(9 * 8).reshape(9, 8)
    assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
    assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
    assert_(np.ma.median(x, axis=1) is not MaskedArray)
    x = np.arange(9 * 8.0).reshape(9, 8)
    assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
    assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
    assert_(np.ma.median(x, axis=1) is not MaskedArray)