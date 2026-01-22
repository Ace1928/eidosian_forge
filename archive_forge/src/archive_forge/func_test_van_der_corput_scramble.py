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
def test_van_der_corput_scramble(self):
    seed = 338213789010180879520345496831675783177
    out = van_der_corput(10, scramble=True, seed=seed)
    sample = van_der_corput(7, start_index=3, scramble=True, seed=seed)
    assert_allclose(sample, out[3:])
    sample = van_der_corput(7, start_index=3, scramble=True, seed=seed, workers=4)
    assert_allclose(sample, out[3:])
    sample = van_der_corput(7, start_index=3, scramble=True, seed=seed, workers=8)
    assert_allclose(sample, out[3:])