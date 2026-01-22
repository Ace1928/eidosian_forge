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
@pytest.mark.filterwarnings('ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.skipif(not pyamg_available, reason='PyAMG is required for the tests in this function.')
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_spectral_embedding_amg_solver(dtype, coo_container, seed=36):
    se_amg = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', eigen_solver='amg', n_neighbors=5, random_state=np.random.RandomState(seed))
    se_arpack = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', eigen_solver='arpack', n_neighbors=5, random_state=np.random.RandomState(seed))
    embed_amg = se_amg.fit_transform(S.astype(dtype))
    embed_arpack = se_arpack.fit_transform(S.astype(dtype))
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-05)
    row = np.array([0, 0, 1, 2, 3, 3, 4], dtype=np.int32)
    col = np.array([1, 2, 2, 3, 4, 5, 5], dtype=np.int32)
    val = np.array([100, 100, 100, 1, 100, 100, 100], dtype=np.int64)
    affinity = coo_container((np.hstack([val, val]), (np.hstack([row, col]), np.hstack([col, row]))), shape=(6, 6))
    se_amg.affinity = 'precomputed'
    se_arpack.affinity = 'precomputed'
    embed_amg = se_amg.fit_transform(affinity.astype(dtype))
    embed_arpack = se_arpack.fit_transform(affinity.astype(dtype))
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-05)
    affinity = affinity.tocsr()
    affinity.indptr = affinity.indptr.astype(np.int64)
    affinity.indices = affinity.indices.astype(np.int64)
    scipy_graph_traversal_supports_int64_index = sp_version >= parse_version('1.11.3')
    if scipy_graph_traversal_supports_int64_index:
        se_amg.fit_transform(affinity)
    else:
        err_msg = 'Only sparse matrices with 32-bit integer indices are accepted'
        with pytest.raises(ValueError, match=err_msg):
            se_amg.fit_transform(affinity)