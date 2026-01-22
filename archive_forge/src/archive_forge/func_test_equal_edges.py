import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_equal_edges(self):
    """ Test that adjacent entries in an edge array can be equal """
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    x_edges = np.array([0, 2, 2])
    y_edges = 1
    hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))
    hist_expected = np.array([[2.0], [1.0]])
    assert_equal(hist, hist_expected)