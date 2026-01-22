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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_answer_gradient_two_points(csr_container):
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0]])
    pos_output = np.array([[-4.961291e-05, -0.0001072243], [9.25946e-05, 0.0002702024]])
    neighbors = np.array([[1], [0]])
    grad_output = np.array([[-2.37012478e-05, -6.29044398e-05], [2.37012478e-05, 6.29044398e-05]])
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, csr_container)