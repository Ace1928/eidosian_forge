import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_1d_without_missing(self):
    x = self.data
    assert_almost_equal(np.corrcoef(x), corrcoef(x))
    assert_almost_equal(np.corrcoef(x, rowvar=False), corrcoef(x, rowvar=False))
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, 'bias and ddof have no effect')
        assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True), corrcoef(x, rowvar=False, bias=True))