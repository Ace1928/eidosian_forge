import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_bool_conversion(self):
    a = np.array([1, 1, 0], dtype=np.uint8)
    int_hist, int_edges = np.histogram(a)
    with suppress_warnings() as sup:
        rec = sup.record(RuntimeWarning, 'Converting input from .*')
        hist, edges = np.histogram([True, True, False])
        assert_equal(len(rec), 1)
        assert_array_equal(hist, int_hist)
        assert_array_equal(edges, int_edges)