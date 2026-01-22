import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_unsigned_monotonicity_check(self):
    arr = np.array([2])
    bins = np.array([1, 3, 1], dtype='uint64')
    with assert_raises(ValueError):
        hist, edges = np.histogram(arr, bins=bins)