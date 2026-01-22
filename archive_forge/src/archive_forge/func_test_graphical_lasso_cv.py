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
def test_graphical_lasso_cv(random_state=1):
    dim = 5
    n_samples = 6
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.96, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    orig_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        GraphicalLassoCV(verbose=100, alphas=5, tol=0.1).fit(X)
    finally:
        sys.stdout = orig_stdout