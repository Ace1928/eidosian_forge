import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_perfect_signal_recovery():
    idx, = gamma[:, 0].nonzero()
    gamma_rec = orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)
    gamma_gram = orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5)
    assert_array_equal(idx, np.flatnonzero(gamma_rec))
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_rec, decimal=2)
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)