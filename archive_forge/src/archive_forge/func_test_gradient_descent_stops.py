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
def test_gradient_descent_stops():

    class ObjectiveSmallGradient:

        def __init__(self):
            self.it = -1

        def __call__(self, _, compute_error=True):
            self.it += 1
            return ((10 - self.it) / 10.0, np.array([1e-05]))

    def flat_function(_, compute_error=True):
        return (0.0, np.ones(1))
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(ObjectiveSmallGradient(), np.zeros(1), 0, n_iter=100, n_iter_without_progress=100, momentum=0.0, learning_rate=0.0, min_gain=0.0, min_grad_norm=1e-05, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert error == 1.0
    assert it == 0
    assert 'gradient norm' in out
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(flat_function, np.zeros(1), 0, n_iter=100, n_iter_without_progress=10, momentum=0.0, learning_rate=0.0, min_gain=0.0, min_grad_norm=0.0, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert error == 0.0
    assert it == 11
    assert 'did not make any progress' in out
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, error, it = _gradient_descent(ObjectiveSmallGradient(), np.zeros(1), 0, n_iter=11, n_iter_without_progress=100, momentum=0.0, learning_rate=0.0, min_gain=0.0, min_grad_norm=0.0, verbose=2)
    finally:
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout
    assert error == 0.0
    assert it == 10
    assert 'Iteration 10' in out