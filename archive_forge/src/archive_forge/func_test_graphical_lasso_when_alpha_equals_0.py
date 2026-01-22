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
def test_graphical_lasso_when_alpha_equals_0():
    """Test graphical_lasso's early return condition when alpha=0."""
    X = np.random.randn(100, 10)
    emp_cov = empirical_covariance(X, assume_centered=True)
    model = GraphicalLasso(alpha=0, covariance='precomputed').fit(emp_cov)
    assert_allclose(model.precision_, np.linalg.inv(emp_cov))
    _, precision = graphical_lasso(emp_cov, alpha=0)
    assert_allclose(precision, np.linalg.inv(emp_cov))