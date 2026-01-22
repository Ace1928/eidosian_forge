import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_width_condition(self):
    """
        Test width condition for peaks.
        """
    x = np.array([1, 0, 1, 2, 1, 0, -1, 4, 0])
    peaks, props = find_peaks(x, width=(None, 2), rel_height=0.75)
    assert_equal(peaks.size, 1)
    assert_equal(peaks, 7)
    assert_allclose(props['widths'], 1.35)
    assert_allclose(props['width_heights'], 1.0)
    assert_allclose(props['left_ips'], 6.4)
    assert_allclose(props['right_ips'], 7.75)