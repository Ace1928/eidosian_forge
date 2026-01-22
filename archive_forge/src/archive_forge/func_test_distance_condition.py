import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_distance_condition(self):
    """
        Test distance condition for peaks.
        """
    peaks_all = np.arange(1, 21, 3)
    x = np.zeros(21)
    x[peaks_all] += np.linspace(1, 2, peaks_all.size)
    assert_equal(find_peaks(x, distance=3)[0], peaks_all)
    peaks_subset = find_peaks(x, distance=3.0001)[0]
    assert_(np.setdiff1d(peaks_subset, peaks_all, assume_unique=True).size == 0)
    assert_equal(np.diff(peaks_subset), 6)
    x = [-2, 1, -1, 0, -3]
    peaks_subset = find_peaks(x, distance=10)[0]
    assert_(peaks_subset.size == 1 and peaks_subset[0] == 1)