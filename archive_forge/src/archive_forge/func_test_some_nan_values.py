import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_some_nan_values(self):
    one_nan = np.array([0, 1, np.nan])
    all_nan = np.array([np.nan, np.nan])
    sup = suppress_warnings()
    sup.filter(RuntimeWarning)
    with sup:
        assert_raises(ValueError, histogram, one_nan, bins='auto')
        assert_raises(ValueError, histogram, all_nan, bins='auto')
        h, b = histogram(one_nan, bins='auto', range=(0, 1))
        assert_equal(h.sum(), 2)
        h, b = histogram(all_nan, bins='auto', range=(0, 1))
        assert_equal(h.sum(), 0)
        h, b = histogram(one_nan, bins=[0, 1])
        assert_equal(h.sum(), 2)
        h, b = histogram(all_nan, bins=[0, 1])
        assert_equal(h.sum(), 0)