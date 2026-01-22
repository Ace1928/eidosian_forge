import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import scipy.optimize._chandrupatla as _chandrupatla
from scipy.optimize._chandrupatla import _chandrupatla_minimize
from itertools import permutations
@np.vectorize
def chandrupatla_single(loc_single):
    return _chandrupatla_minimize(self.f, -5, 0, 5, args=(loc_single,))