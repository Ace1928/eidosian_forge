import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_find_peaks_nopeak(self):
    """
        Verify that no peak is found in
        data that's just noise.
        """
    noise_amp = 1.0
    num_points = 100
    np.random.seed(181819141)
    test_data = (np.random.rand(num_points) - 0.5) * (2 * noise_amp)
    widths = np.arange(10, 50)
    found_locs = find_peaks_cwt(test_data, widths, min_snr=5, noise_perc=30)
    np.testing.assert_equal(len(found_locs), 0)