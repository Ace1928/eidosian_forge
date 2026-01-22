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
def test_spectral_embedding_first_eigen_vector():
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    n_components = 2
    for seed in range(10):
        embedding = spectral_embedding(sims, norm_laplacian=False, n_components=n_components, drop_first=False, random_state=seed)
        assert np.std(embedding[:, 0]) == pytest.approx(0)
        assert np.std(embedding[:, 1]) > 0.001