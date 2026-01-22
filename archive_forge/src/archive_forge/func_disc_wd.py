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
def disc_wd(x):
    n, s = x.shape
    xij = x[None, :, :]
    xkj = x[:, None, :]
    disc = np.sum(np.sum(np.prod(3 / 2 - np.abs(xij - xkj) + np.abs(xij - xkj) ** 2, axis=2), axis=0))
    return -(4 / 3) ** s + 1 / n ** 2 * disc