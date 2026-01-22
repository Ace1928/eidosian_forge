import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_limited_variance(self):
    """
        Check when IQR is 0, but variance exists, we return the sturges value
        and not the fd value.
        """
    lim_var_data = np.ones(1000)
    lim_var_data[:3] = 0
    lim_var_data[-4:] = 100
    edges_auto = histogram_bin_edges(lim_var_data, 'auto')
    assert_equal(edges_auto, np.linspace(0, 100, 12))
    edges_fd = histogram_bin_edges(lim_var_data, 'fd')
    assert_equal(edges_fd, np.array([0, 100]))
    edges_sturges = histogram_bin_edges(lim_var_data, 'sturges')
    assert_equal(edges_sturges, np.linspace(0, 100, 12))