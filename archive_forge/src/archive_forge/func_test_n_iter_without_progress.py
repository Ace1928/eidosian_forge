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
@skip_if_32bit
def test_n_iter_without_progress():
    random_state = check_random_state(0)
    X = random_state.randn(100, 10)
    for method in ['barnes_hut', 'exact']:
        tsne = TSNE(n_iter_without_progress=-1, verbose=2, learning_rate=100000000.0, random_state=0, method=method, n_iter=351, init='random')
        tsne._N_ITER_CHECK = 1
        tsne._EXPLORATION_N_ITER = 0
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            tsne.fit_transform(X)
        finally:
            out = sys.stdout.getvalue()
            sys.stdout.close()
            sys.stdout = old_stdout
        assert 'did not make any progress during the last -1 episodes. Finished.' in out