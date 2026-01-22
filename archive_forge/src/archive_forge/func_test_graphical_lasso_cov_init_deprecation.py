import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from sklearn import datasets
from sklearn.covariance import (
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_graphical_lasso_cov_init_deprecation():
    """Check that we raise a deprecation warning if providing `cov_init` in
    `graphical_lasso`."""
    rng, dim, n_samples = (np.random.RandomState(0), 20, 100)
    prec = make_sparse_spd_matrix(dim, alpha=0.95, random_state=0)
    cov = linalg.inv(prec)
    X = rng.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    emp_cov = empirical_covariance(X)
    with pytest.warns(FutureWarning, match='cov_init parameter is deprecated'):
        graphical_lasso(emp_cov, alpha=0.1, cov_init=emp_cov)