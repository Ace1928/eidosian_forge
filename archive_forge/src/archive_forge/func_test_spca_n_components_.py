import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('SPCA', [SparsePCA, MiniBatchSparsePCA])
@pytest.mark.parametrize('n_components', [None, 3])
def test_spca_n_components_(SPCA, n_components):
    rng = np.random.RandomState(0)
    n_samples, n_features = (12, 10)
    X = rng.randn(n_samples, n_features)
    model = SPCA(n_components=n_components).fit(X)
    if n_components is not None:
        assert model.n_components_ == n_components
    else:
        assert model.n_components_ == n_features