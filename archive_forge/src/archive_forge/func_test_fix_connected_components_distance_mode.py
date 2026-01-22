import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import _fix_connected_components
def test_fix_connected_components_distance_mode():
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode='distance')
    assert np.all(graph.data == 1)
    n_connected_components, labels = connected_components(graph)
    graph = _fix_connected_components(X, graph, n_connected_components, labels, mode='distance')
    assert not np.all(graph.data == 1)