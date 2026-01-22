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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_fit_transform_csr_matrix(method, csr_container):
    rng = check_random_state(0)
    X = rng.randn(50, 2)
    X[rng.randint(0, 50, 25), rng.randint(0, 2, 25)] = 0.0
    X_csr = csr_container(X)
    tsne = TSNE(n_components=2, init='random', perplexity=10, learning_rate=100.0, random_state=0, method=method, n_iter=750)
    X_embedded = tsne.fit_transform(X_csr)
    assert_allclose(trustworthiness(X_csr, X_embedded, n_neighbors=1), 1.0, rtol=0.11)