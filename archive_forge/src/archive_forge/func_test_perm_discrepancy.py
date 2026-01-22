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
def test_perm_discrepancy(self):
    rng = np.random.default_rng(46449423132557934943847369749645759997)
    qmc_gen = qmc.LatinHypercube(5, seed=rng)
    sample = qmc_gen.random(10)
    disc = qmc.discrepancy(sample)
    for i in range(100):
        row_1 = rng.integers(10)
        row_2 = rng.integers(10)
        col = rng.integers(5)
        disc = _perturb_discrepancy(sample, row_1, row_2, col, disc)
        sample[row_1, col], sample[row_2, col] = (sample[row_2, col], sample[row_1, col])
        disc_reference = qmc.discrepancy(sample)
        assert_allclose(disc, disc_reference)