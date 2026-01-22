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
def test_graphical_lasso_cv_scores():
    splits = 4
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array([[0.8, 0.0, 0.2, 0.0], [0.0, 0.4, 0.0, 0.0], [0.2, 0.0, 0.3, 0.1], [0.0, 0.0, 0.1, 0.7]])
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    cov = GraphicalLassoCV(cv=splits, alphas=n_alphas, n_refinements=n_refinements).fit(X)
    cv_results = cov.cv_results_
    total_alphas = n_refinements * n_alphas + 1
    keys = ['alphas']
    split_keys = [f'split{i}_test_score' for i in range(splits)]
    for key in keys + split_keys:
        assert key in cv_results
        assert len(cv_results[key]) == total_alphas
    cv_scores = np.asarray([cov.cv_results_[key] for key in split_keys])
    expected_mean = cv_scores.mean(axis=0)
    expected_std = cv_scores.std(axis=0)
    assert_allclose(cov.cv_results_['mean_test_score'], expected_mean)
    assert_allclose(cov.cv_results_['std_test_score'], expected_std)