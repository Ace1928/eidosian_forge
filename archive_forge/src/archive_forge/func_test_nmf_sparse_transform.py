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
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_nmf_sparse_transform(Estimator, solver, csc_container):
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(3, 2))
    A[1, 1] = 0
    A = csc_container(A)
    model = Estimator(random_state=0, n_components=2, max_iter=400, **solver)
    A_fit_tr = model.fit_transform(A)
    A_tr = model.transform(A)
    assert_allclose(A_fit_tr, A_tr, atol=0.1)