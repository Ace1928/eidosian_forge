import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_bin_edge_cases(self):
    arr = np.array([337, 404, 739, 806, 1007, 1811, 2012])
    hist, edges = np.histogram(arr, bins=8296, range=(2, 2280))
    mask = hist > 0
    left_edges = edges[:-1][mask]
    right_edges = edges[1:][mask]
    for x, left, right in zip(arr, left_edges, right_edges):
        assert_(x >= left)
        assert_(x < right)