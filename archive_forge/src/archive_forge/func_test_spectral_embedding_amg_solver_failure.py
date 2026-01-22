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
@pytest.mark.filterwarnings('ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.filterwarnings('ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*')
@pytest.mark.filterwarnings('ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.skipif(not pyamg_available, reason='PyAMG is required for the tests in this function.')
@pytest.mark.filterwarnings('ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_spectral_embedding_amg_solver_failure(dtype, seed=36):
    num_nodes = 100
    X = sparse.rand(num_nodes, num_nodes, density=0.1, random_state=seed)
    X = X.astype(dtype)
    upper = sparse.triu(X) - sparse.diags(X.diagonal())
    sym_matrix = upper + upper.T
    embedding = spectral_embedding(sym_matrix, n_components=10, eigen_solver='amg', random_state=0)
    for i in range(3):
        new_embedding = spectral_embedding(sym_matrix, n_components=10, eigen_solver='amg', random_state=i + 1)
        _assert_equal_with_sign_flipping(embedding, new_embedding, tol=0.05)