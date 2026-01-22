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
def test_spectral_embedding_two_components(eigen_solver, dtype, seed=0):
    random_state = np.random.RandomState(seed)
    n_sample = 100
    affinity = np.zeros(shape=[n_sample * 2, n_sample * 2])
    affinity[0:n_sample, 0:n_sample] = np.abs(random_state.randn(n_sample, n_sample)) + 2
    affinity[n_sample:, n_sample:] = np.abs(random_state.randn(n_sample, n_sample)) + 2
    component = _graph_connected_component(affinity, 0)
    assert component[:n_sample].all()
    assert not component[n_sample:].any()
    component = _graph_connected_component(affinity, -1)
    assert not component[:n_sample].any()
    assert component[n_sample:].all()
    affinity[0, n_sample + 1] = 1
    affinity[n_sample + 1, 0] = 1
    affinity.flat[::2 * n_sample + 1] = 0
    affinity = 0.5 * (affinity + affinity.T)
    true_label = np.zeros(shape=2 * n_sample)
    true_label[0:n_sample] = 1
    se_precomp = SpectralEmbedding(n_components=1, affinity='precomputed', random_state=np.random.RandomState(seed), eigen_solver=eigen_solver)
    embedded_coordinate = se_precomp.fit_transform(affinity.astype(dtype))
    label_ = np.array(embedded_coordinate.ravel() < 0, dtype=np.int64)
    assert normalized_mutual_info_score(true_label, label_) == pytest.approx(1.0)