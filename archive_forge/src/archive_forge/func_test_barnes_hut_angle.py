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
def test_barnes_hut_angle():
    angle = 0.0
    perplexity = 10
    n_samples = 100
    for n_components in [2, 3]:
        n_features = 5
        degrees_of_freedom = float(n_components - 1.0)
        random_state = check_random_state(0)
        data = random_state.randn(n_samples, n_features)
        distances = pairwise_distances(data)
        params = random_state.randn(n_samples, n_components)
        P = _joint_probabilities(distances, perplexity, verbose=0)
        kl_exact, grad_exact = _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components)
        n_neighbors = n_samples - 1
        distances_csr = NearestNeighbors().fit(data).kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
        P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
        kl_bh, grad_bh = _kl_divergence_bh(params, P_bh, degrees_of_freedom, n_samples, n_components, angle=angle, skip_num_points=0, verbose=0)
        P = squareform(P)
        P_bh = P_bh.toarray()
        assert_array_almost_equal(P_bh, P, decimal=5)
        assert_almost_equal(kl_exact, kl_bh, decimal=3)