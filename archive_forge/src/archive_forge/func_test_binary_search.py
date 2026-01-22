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
def test_binary_search():
    random_state = check_random_state(0)
    data = random_state.randn(50, 5)
    distances = pairwise_distances(data).astype(np.float32)
    desired_perplexity = 25.0
    P = _binary_search_perplexity(distances, desired_perplexity, verbose=0)
    P = np.maximum(P, np.finfo(np.double).eps)
    mean_perplexity = np.mean([np.exp(-np.sum(P[i] * np.log(P[i]))) for i in range(P.shape[0])])
    assert_almost_equal(mean_perplexity, desired_perplexity, decimal=3)