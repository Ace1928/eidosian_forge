import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_plateau_size(self):
    """
        Test plateau size condition for peaks.
        """
    plateau_sizes = np.array([1, 2, 3, 4, 8, 20, 111])
    x = np.zeros(plateau_sizes.size * 2 + 1)
    x[1::2] = plateau_sizes
    repeats = np.ones(x.size, dtype=int)
    repeats[1::2] = x[1::2]
    x = np.repeat(x, repeats)
    peaks, props = find_peaks(x, plateau_size=(None, None))
    assert_equal(peaks, [1, 3, 7, 11, 18, 33, 100])
    assert_equal(props['plateau_sizes'], plateau_sizes)
    assert_equal(props['left_edges'], peaks - (plateau_sizes - 1) // 2)
    assert_equal(props['right_edges'], peaks + plateau_sizes // 2)
    assert_equal(find_peaks(x, plateau_size=4)[0], [11, 18, 33, 100])
    assert_equal(find_peaks(x, plateau_size=(None, 3.5))[0], [1, 3, 7])
    assert_equal(find_peaks(x, plateau_size=(5, 50))[0], [18, 33])