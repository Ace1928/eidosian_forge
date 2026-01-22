import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_incorrect_methods(self):
    """
        Check a Value Error is thrown when an unknown string is passed in
        """
    check_list = ['mad', 'freeman', 'histograms', 'IQR']
    for estimator in check_list:
        assert_raises(ValueError, histogram, [1, 2, 3], estimator)