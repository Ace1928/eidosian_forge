import sys
from io import StringIO
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from scipy.spatial.distance import pdist, squareform
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.exceptions import EfficiencyWarning
from sklearn.manifold import (  # type: ignore
from sklearn.manifold._t_sne import (
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics.pairwise import (
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
def test_binary_perplexity_stability():
    n_neighbors = 10
    n_samples = 100
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, 5)
    nn = NearestNeighbors().fit(data)
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
    distances = distance_graph.data.astype(np.float32, copy=False)
    distances = distances.reshape(n_samples, n_neighbors)
    last_P = None
    desired_perplexity = 3
    for _ in range(100):
        P = _binary_search_perplexity(distances.copy(), desired_perplexity, verbose=0)
        P1 = _joint_probabilities_nn(distance_graph, desired_perplexity, verbose=0)
        P1 = P1.toarray()
        if last_P is None:
            last_P = P
            last_P1 = P1
        else:
            assert_array_almost_equal(P, last_P, decimal=4)
            assert_array_almost_equal(P1, last_P1, decimal=4)