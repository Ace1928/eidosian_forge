import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_find_peaks_exact(self):
    """
        Generate a series of gaussians and attempt to find the peak locations.
        """
    sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
    num_points = 500
    test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
    widths = np.arange(0.1, max(sigmas))
    found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=0, min_length=None)
    np.testing.assert_array_equal(found_locs, act_locs, 'Found maximum locations did not equal those expected')