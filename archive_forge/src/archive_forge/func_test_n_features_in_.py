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
@pytest.mark.parametrize('est', (SpectralBiclustering(), SpectralCoclustering()))
def test_n_features_in_(est):
    X, _, _ = make_biclusters((3, 3), 3, random_state=0)
    assert not hasattr(est, 'n_features_in_')
    est.fit(X)
    assert est.n_features_in_ == 3