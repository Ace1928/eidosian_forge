import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb
from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement
def sample_without_replacement_method(n_population, n_samples, random_state=None):
    return sample_without_replacement(n_population, n_samples, method=m, random_state=random_state)