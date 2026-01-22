import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_silhouette_samples_euclidean_sparse(sparse_container):
    """Check that silhouette_samples works for sparse matrices correctly."""
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pdist_dense = pairwise_distances(X)
    pdist_sparse = sparse_container(pdist_dense)
    assert issparse(pdist_sparse)
    output_with_sparse_input = silhouette_samples(pdist_sparse, y)
    output_with_dense_input = silhouette_samples(pdist_dense, y)
    assert_allclose(output_with_sparse_input, output_with_dense_input)