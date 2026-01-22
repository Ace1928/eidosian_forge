import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_wlen(self):
    """
        Test if wlen actually shrinks the evaluation range correctly.
        """
    x = [0, 1, 2, 3, 1, 0, -1]
    peak = [3]
    assert_equal(peak_prominences(x, peak), [3.0, 0, 6])
    for wlen, i in [(8, 0), (7, 0), (6, 0), (5, 1), (3.2, 1), (3, 2), (1.1, 2)]:
        assert_equal(peak_prominences(x, peak, wlen), [3.0 - i, 0 + i, 6 - i])