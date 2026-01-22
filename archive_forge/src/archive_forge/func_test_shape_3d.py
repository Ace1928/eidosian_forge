import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_shape_3d(self):
    bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4), (4, 5, 6))
    r = np.random.rand(10, 3)
    for b in bins:
        H, edges = histogramdd(r, b)
        assert_(H.shape == b)