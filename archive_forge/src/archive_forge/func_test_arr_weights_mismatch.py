import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_arr_weights_mismatch(self):
    a = np.arange(10) + 0.5
    w = np.arange(11) + 0.5
    with assert_raises_regex(ValueError, 'same shape as'):
        h, b = histogram(a, range=[1, 9], weights=w, density=True)