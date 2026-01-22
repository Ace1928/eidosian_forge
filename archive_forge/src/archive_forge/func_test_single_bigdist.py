import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def test_single_bigdist(self):
    distances = [0, 1, 2, 5]
    gaps = [0, 1, 2, 4]
    test_matr = np.zeros([20, 50])
    length = 12
    line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
    test_matr[line[0], line[1]] = 1
    max_dist = 3
    max_distances = np.full(20, max_dist)
    identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
    assert_(len(identified_lines) == 2)
    for iline in identified_lines:
        adists = np.diff(iline[1])
        np.testing.assert_array_less(np.abs(adists), max_dist)
        agaps = np.diff(iline[0])
        np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)