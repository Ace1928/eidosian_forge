import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
@pytest.mark.parametrize('x', [np.array([1.0, 0, 2]), np.array([3.0, 3, 0, 4, 4]), np.array([5.0, 5, 5, 0, 6, 6, 6])])
def test_signal_edges(self, x):
    """Test if behavior on signal edges is correct."""
    for array in _local_maxima_1d(x):
        assert_equal(array, np.array([]))
        assert_(array.base is None)