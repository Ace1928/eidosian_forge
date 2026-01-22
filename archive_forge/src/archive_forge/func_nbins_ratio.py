import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def nbins_ratio(seed, size):
    rng = np.random.RandomState(seed)
    x = rng.normal(loc=0, scale=2, size=size)
    a, b = (len(np.histogram(x, 'stone')[0]), len(np.histogram(x, 'scott')[0]))
    return a / (a + b)