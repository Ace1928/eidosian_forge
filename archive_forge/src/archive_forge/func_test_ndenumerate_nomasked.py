import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_ndenumerate_nomasked(self):
    ordinary = np.arange(6.0).reshape((1, 3, 2))
    empty_mask = np.zeros_like(ordinary, dtype=bool)
    with_mask = masked_array(ordinary, mask=empty_mask)
    assert_equal(list(np.ndenumerate(ordinary)), list(ndenumerate(ordinary)))
    assert_equal(list(ndenumerate(ordinary)), list(ndenumerate(with_mask)))
    assert_equal(list(ndenumerate(with_mask)), list(ndenumerate(with_mask, compressed=False)))