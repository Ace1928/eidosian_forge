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
@pytest.mark.parametrize('method', ['exact', 'barnes_hut'])
def test_tsne_n_jobs(method):
    """Make sure that the n_jobs parameter doesn't impact the output"""
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features)
    X_tr_ref = TSNE(n_components=2, method=method, perplexity=25.0, angle=0, n_jobs=1, random_state=0, init='random', learning_rate='auto').fit_transform(X)
    X_tr = TSNE(n_components=2, method=method, perplexity=25.0, angle=0, n_jobs=2, random_state=0, init='random', learning_rate='auto').fit_transform(X)
    assert_allclose(X_tr_ref, X_tr)