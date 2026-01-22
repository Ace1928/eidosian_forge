import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_one_bin(self):
    hist, edges = histogram([1, 2, 3, 4], [1, 2])
    assert_array_equal(hist, [2])
    assert_array_equal(edges, [1, 2])
    assert_raises(ValueError, histogram, [1, 2], bins=0)
    h, e = histogram([1, 2], bins=1)
    assert_equal(h, np.array([2]))
    assert_allclose(e, np.array([1.0, 2.0]))