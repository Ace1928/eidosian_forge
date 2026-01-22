import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_gh_23110(self):
    hist, e = np.histogram(np.array([-9e-309], dtype='>f8'), bins=2, range=(-1e-308, -2e-313))
    expected_hist = np.array([1, 0])
    assert_array_equal(hist, expected_hist)