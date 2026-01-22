import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def do_signed_overflow_bounds(self, dtype):
    exponent = 8 * np.dtype(dtype).itemsize - 1
    arr = np.array([-2 ** exponent + 4, 2 ** exponent - 4], dtype=dtype)
    hist, e = histogram(arr, bins=2)
    assert_equal(e, [-2 ** exponent + 4, 0, 2 ** exponent - 4])
    assert_equal(hist, [1, 1])