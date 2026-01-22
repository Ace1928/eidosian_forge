import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_minimal(self):
    test_matr = np.zeros([20, 100])
    test_matr[0, 10] = 1
    lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
    assert_(len(lines) == 1)
    test_matr = np.zeros([20, 100])
    test_matr[0:2, 10] = 1
    lines = _identify_ridge_lines(test_matr, np.full(20, 2), 1)
    assert_(len(lines) == 1)