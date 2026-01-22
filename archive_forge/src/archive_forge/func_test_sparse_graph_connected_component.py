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
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_sparse_graph_connected_component(coo_container):
    rng = np.random.RandomState(42)
    n_samples = 300
    boundaries = [0, 42, 121, 200, n_samples]
    p = rng.permutation(n_samples)
    connections = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group = p[start:stop]
        for i in range(len(group) - 1):
            connections.append((group[i], group[i + 1]))
        min_idx, max_idx = (0, len(group) - 1)
        n_random_connections = 1000
        source = rng.randint(min_idx, max_idx, size=n_random_connections)
        target = rng.randint(min_idx, max_idx, size=n_random_connections)
        connections.extend(zip(group[source], group[target]))
    row_idx, column_idx = tuple(np.array(connections).T)
    data = rng.uniform(0.1, 42, size=len(connections))
    affinity = coo_container((data, (row_idx, column_idx)))
    affinity = 0.5 * (affinity + affinity.T)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        component_1 = _graph_connected_component(affinity, p[start])
        component_size = stop - start
        assert component_1.sum() == component_size
        component_2 = _graph_connected_component(affinity, p[stop - 1])
        assert component_2.sum() == component_size
        assert_array_equal(component_1, component_2)