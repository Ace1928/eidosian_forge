import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import _fix_connected_components
def test_fix_connected_components_precomputed():
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode='distance')
    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1
    distances = pairwise_distances(X)
    graph = _fix_connected_components(distances, graph, n_connected_components, labels, metric='precomputed')
    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1
    with pytest.raises(RuntimeError, match='does not work with a sparse'):
        _fix_connected_components(graph, graph, n_connected_components, labels, metric='precomputed')