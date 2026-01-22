import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_FindIndex(self):
    p_cumulative = np.array([0.1, 0.4, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0])
    size = len(p_cumulative)
    assert_equal(_test_find_index(p_cumulative, size, 0.0), 0)
    assert_equal(_test_find_index(p_cumulative, size, 0.4), 2)
    assert_equal(_test_find_index(p_cumulative, size, 0.44999), 2)
    assert_equal(_test_find_index(p_cumulative, size, 0.45001), 3)
    assert_equal(_test_find_index(p_cumulative, size, 1.0), size - 1)