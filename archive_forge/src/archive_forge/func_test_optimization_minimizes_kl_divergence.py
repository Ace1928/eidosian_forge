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
def test_optimization_minimizes_kl_divergence():
    """t-SNE should give a lower KL divergence with more iterations."""
    random_state = check_random_state(0)
    X, _ = make_blobs(n_features=3, random_state=random_state)
    kl_divergences = []
    for n_iter in [250, 300, 350]:
        tsne = TSNE(n_components=2, init='random', perplexity=10, learning_rate=100.0, n_iter=n_iter, random_state=0)
        tsne.fit_transform(X)
        kl_divergences.append(tsne.kl_divergence_)
    assert kl_divergences[1] <= kl_divergences[0]
    assert kl_divergences[2] <= kl_divergences[1]