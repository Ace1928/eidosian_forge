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
def test_get_submatrix(csr_container):
    data = np.arange(20).reshape(5, 4)
    model = MockBiclustering()
    for X in (data, csr_container(data), data.tolist()):
        submatrix = model.get_submatrix(0, X)
        if issparse(submatrix):
            submatrix = submatrix.toarray()
        assert_array_equal(submatrix, [[2, 3], [6, 7], [18, 19]])
        submatrix[:] = -1
        if issparse(X):
            X = X.toarray()
        assert np.all(X != -1)