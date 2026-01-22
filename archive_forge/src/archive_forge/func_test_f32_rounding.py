import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_f32_rounding(self):
    x = np.array([276.318359, -69.593948, 21.329449], dtype=np.float32)
    y = np.array([5005.689453, 4481.327637, 6010.369629], dtype=np.float32)
    counts_hist, xedges, yedges = np.histogram2d(x, y, bins=100)
    assert_equal(counts_hist.sum(), 3.0)