import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def do_precision_lower_bound(self, float_small, float_large):
    eps = np.finfo(float_large).eps
    arr = np.array([1.0], float_small)
    range = np.array([1.0 + eps, 2.0], float_large)
    if range.astype(float_small)[0] != 1:
        return
    count, x_loc = np.histogram(arr, bins=1, range=range)
    assert_equal(count, [1])
    assert_equal(x_loc.dtype, float_small)