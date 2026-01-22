import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_novariance(self):
    """
        Check that methods handle no variance in data
        Primarily for Scott and FD as the SD and IQR are both 0 in this case
        """
    novar_dataset = np.ones(100)
    novar_resultdict = {'fd': 1, 'scott': 1, 'rice': 1, 'sturges': 1, 'doane': 1, 'sqrt': 1, 'auto': 1, 'stone': 1}
    for estimator, numbins in novar_resultdict.items():
        a, b = np.histogram(novar_dataset, estimator)
        assert_equal(len(a), numbins, err_msg='{0} estimator, No Variance test'.format(estimator))