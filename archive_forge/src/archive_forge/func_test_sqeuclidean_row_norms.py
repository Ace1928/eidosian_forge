import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sqeuclidean_row_norms(global_random_seed, dtype, csr_container):
    rng = np.random.RandomState(global_random_seed)
    spread = 100
    n_samples = rng.choice([97, 100, 101, 1000])
    n_features = rng.choice([5, 10, 100])
    num_threads = rng.choice([1, 2, 8])
    X = rng.rand(n_samples, n_features).astype(dtype) * spread
    X_csr = csr_container(X)
    sq_row_norm_reference = np.linalg.norm(X, axis=1) ** 2
    sq_row_norm = sqeuclidean_row_norms(X, num_threads=num_threads)
    sq_row_norm_csr = sqeuclidean_row_norms(X_csr, num_threads=num_threads)
    assert_allclose(sq_row_norm_reference, sq_row_norm)
    assert_allclose(sq_row_norm_reference, sq_row_norm_csr)
    with pytest.raises(ValueError):
        X = np.asfortranarray(X)
        sqeuclidean_row_norms(X, num_threads=num_threads)