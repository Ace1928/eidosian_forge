import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_spectral_coclustering(global_random_seed, csr_container):
    param_grid = {'svd_method': ['randomized', 'arpack'], 'n_svd_vecs': [None, 20], 'mini_batch': [False, True], 'init': ['k-means++'], 'n_init': [10]}
    S, rows, cols = make_biclusters((30, 30), 3, noise=0.1, random_state=global_random_seed)
    S -= S.min()
    S = np.where(S < 1, 0, S)
    for mat in (S, csr_container(S)):
        for kwargs in ParameterGrid(param_grid):
            model = SpectralCoclustering(n_clusters=3, random_state=global_random_seed, **kwargs)
            model.fit(mat)
            assert model.rows_.shape == (3, 30)
            assert_array_equal(model.rows_.sum(axis=0), np.ones(30))
            assert_array_equal(model.columns_.sum(axis=0), np.ones(30))
            assert consensus_score(model.biclusters_, (rows, cols)) == 1
            _test_shape_indices(model)