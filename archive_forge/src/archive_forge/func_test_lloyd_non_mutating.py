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
def test_lloyd_non_mutating(self):
    """
        Verify that the input samples are not mutated in place and that they do
        not share memory with the output.
        """
    sample_orig = np.array([[0.1, 0.1], [0.1, 0.2], [0.2, 0.1], [0.2, 0.2]])
    sample_copy = sample_orig.copy()
    new_sample = _lloyd_centroidal_voronoi_tessellation(sample=sample_orig)
    assert_allclose(sample_orig, sample_copy)
    assert not np.may_share_memory(sample_orig, new_sample)