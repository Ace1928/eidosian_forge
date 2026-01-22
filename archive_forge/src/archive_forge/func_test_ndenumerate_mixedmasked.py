import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_ndenumerate_mixedmasked(self):
    a = masked_array(np.arange(12).reshape((3, 4)), mask=[[1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
    items = [((1, 2), 6), ((2, 0), 8), ((2, 1), 9), ((2, 2), 10), ((2, 3), 11)]
    assert_equal(list(ndenumerate(a)), items)
    assert_equal(len(list(ndenumerate(a, compressed=False))), a.size)
    for coordinate, value in ndenumerate(a, compressed=False):
        assert_equal(a[coordinate], value)