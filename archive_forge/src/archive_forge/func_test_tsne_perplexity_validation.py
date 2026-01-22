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
@pytest.mark.parametrize('perplexity', (20, 30))
def test_tsne_perplexity_validation(perplexity):
    """Make sure that perplexity > n_samples results in a ValueError"""
    random_state = check_random_state(0)
    X = random_state.randn(20, 2)
    est = TSNE(learning_rate='auto', init='pca', perplexity=perplexity, random_state=random_state)
    msg = 'perplexity must be less than n_samples'
    with pytest.raises(ValueError, match=msg):
        est.fit_transform(X)