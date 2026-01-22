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
@pytest.mark.filterwarnings('ignore:the behavior of nmi will change in version 0.22')
def test_pipeline_spectral_clustering(seed=36):
    random_state = np.random.RandomState(seed)
    se_rbf = SpectralEmbedding(n_components=n_clusters, affinity='rbf', random_state=random_state)
    se_knn = SpectralEmbedding(n_components=n_clusters, affinity='nearest_neighbors', n_neighbors=5, random_state=random_state)
    for se in [se_rbf, se_knn]:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        km.fit(se.fit_transform(S))
        assert_array_almost_equal(normalized_mutual_info_score(km.labels_, true_labels), 1.0, 2)