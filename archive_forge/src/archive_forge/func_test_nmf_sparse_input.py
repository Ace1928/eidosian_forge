import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize(['Estimator', 'solver'], [[NMF, {'solver': 'cd'}], [NMF, {'solver': 'mu'}], [MiniBatchNMF, {}]])
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize('alpha_W', (0.0, 1.0))
@pytest.mark.parametrize('alpha_H', (0.0, 1.0, 'same'))
def test_nmf_sparse_input(Estimator, solver, sparse_container, alpha_W, alpha_H):
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = sparse_container(A)
    est1 = Estimator(n_components=5, init='random', alpha_W=alpha_W, alpha_H=alpha_H, random_state=0, tol=0, max_iter=100, **solver)
    est2 = clone(est1)
    W1 = est1.fit_transform(A)
    W2 = est2.fit_transform(A_sparse)
    H1 = est1.components_
    H2 = est2.components_
    assert_allclose(W1, W2)
    assert_allclose(H1, H2)