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
def test_graphical_lassos(random_state=1):
    """Test the graphical lasso solvers.

    This checks is unstable for some random seeds where the covariance found with "cd"
    and "lars" solvers are different (4 cases / 100 tries).
    """
    dim = 20
    n_samples = 100
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.95, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    emp_cov = empirical_covariance(X)
    for alpha in (0.0, 0.1, 0.25):
        covs = dict()
        icovs = dict()
        for method in ('cd', 'lars'):
            cov_, icov_, costs = graphical_lasso(emp_cov, return_costs=True, alpha=alpha, mode=method)
            covs[method] = cov_
            icovs[method] = icov_
            costs, dual_gap = np.array(costs).T
            if not alpha == 0:
                assert_array_less(np.diff(costs), 1e-12)
        assert_allclose(covs['cd'], covs['lars'], atol=0.0005)
        assert_allclose(icovs['cd'], icovs['lars'], atol=0.0005)
    model = GraphicalLasso(alpha=0.25).fit(X)
    model.score(X)
    assert_array_almost_equal(model.covariance_, covs['cd'], decimal=4)
    assert_array_almost_equal(model.covariance_, covs['lars'], decimal=4)
    Z = X - X.mean(0)
    precs = list()
    for assume_centered in (False, True):
        prec_ = GraphicalLasso(assume_centered=assume_centered).fit(Z).precision_
        precs.append(prec_)
    assert_array_almost_equal(precs[0], precs[1])