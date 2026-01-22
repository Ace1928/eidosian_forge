import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_height_condition(self):
    """
        Test height condition for peaks.
        """
    x = (0.0, 1 / 3, 0.0, 2.5, 0, 4.0, 0)
    peaks, props = find_peaks(x, height=(None, None))
    assert_equal(peaks, np.array([1, 3, 5]))
    assert_equal(props['peak_heights'], np.array([1 / 3, 2.5, 4.0]))
    assert_equal(find_peaks(x, height=0.5)[0], np.array([3, 5]))
    assert_equal(find_peaks(x, height=(None, 3))[0], np.array([1, 3]))
    assert_equal(find_peaks(x, height=(2, 3))[0], np.array([3]))