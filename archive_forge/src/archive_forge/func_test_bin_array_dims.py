import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_bin_array_dims(self):
    vals = np.linspace(0.0, 1.0, num=100)
    bins = np.array([[0, 0.5], [0.6, 1.0]])
    with assert_raises_regex(ValueError, 'must be 1d'):
        np.histogram(vals, bins=bins)