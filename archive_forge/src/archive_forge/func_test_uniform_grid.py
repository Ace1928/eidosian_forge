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
@pytest.mark.parametrize('method', ['barnes_hut', 'exact'])
def test_uniform_grid(method):
    """Make sure that TSNE can approximately recover a uniform 2D grid

    Due to ties in distances between point in X_2d_grid, this test is platform
    dependent for ``method='barnes_hut'`` due to numerical imprecision.

    Also, t-SNE is not assured to converge to the right solution because bad
    initialization can lead to convergence to bad local minimum (the
    optimization problem is non-convex). To avoid breaking the test too often,
    we re-run t-SNE from the final point when the convergence is not good
    enough.
    """
    seeds = range(3)
    n_iter = 500
    for seed in seeds:
        tsne = TSNE(n_components=2, init='random', random_state=seed, perplexity=50, n_iter=n_iter, method=method, learning_rate='auto')
        Y = tsne.fit_transform(X_2d_grid)
        try_name = '{}_{}'.format(method, seed)
        try:
            assert_uniform_grid(Y, try_name)
        except AssertionError:
            try_name += ':rerun'
            tsne.init = Y
            Y = tsne.fit_transform(X_2d_grid)
            assert_uniform_grid(Y, try_name)