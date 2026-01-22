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
@pytest.mark.parametrize('init', ('random', 'pca'))
def test_preserve_trustworthiness_approximately(method, init):
    random_state = check_random_state(0)
    n_components = 2
    X = random_state.randn(50, n_components).astype(np.float32)
    tsne = TSNE(n_components=n_components, init=init, random_state=0, method=method, n_iter=700, learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    t = trustworthiness(X, X_embedded, n_neighbors=1)
    assert t > 0.85