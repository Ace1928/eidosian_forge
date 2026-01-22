from unittest.mock import Mock
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding, _spectral_embedding, spectral_embedding
from sklearn.manifold._spectral_embedding import (
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import (
from sklearn.utils.fixes import laplacian as csgraph_laplacian
@pytest.mark.parametrize('eigen_solver', ['arpack', 'lobpcg', pytest.param('amg', marks=skip_if_no_pyamg)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_spectral_embedding_preserves_dtype(eigen_solver, dtype):
    """Check that `SpectralEmbedding is preserving the dtype of the fitted
    attribute and transformed data.

    Ideally, this test should be covered by the common test
    `check_transformer_preserve_dtypes`. However, this test only run
    with transformers implementing `transform` while `SpectralEmbedding`
    implements only `fit_transform`.
    """
    X = S.astype(dtype)
    se = SpectralEmbedding(n_components=2, affinity='rbf', eigen_solver=eigen_solver, random_state=0)
    X_trans = se.fit_transform(X)
    assert X_trans.dtype == dtype
    assert se.embedding_.dtype == dtype
    assert se.affinity_matrix_.dtype == dtype