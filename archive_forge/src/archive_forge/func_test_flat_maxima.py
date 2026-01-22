import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_flat_maxima(self):
    """Test if flat maxima are detected correctly."""
    x = np.array([-1.3, 0, 1, 0, 2, 2, 0, 3, 3, 3, 2.99, 4, 4, 4, 4, -10, -5, -5, -5, -5, -5, -10])
    midpoints, left_edges, right_edges = _local_maxima_1d(x)
    assert_equal(midpoints, np.array([2, 4, 8, 12, 18]))
    assert_equal(left_edges, np.array([2, 4, 7, 11, 16]))
    assert_equal(right_edges, np.array([2, 5, 9, 14, 20]))