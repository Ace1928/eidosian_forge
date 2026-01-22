import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
@pytest.mark.parametrize('Est', (PLSRegression, PLSCanonical, CCA))
def test_univariate_equivalence(Est):
    d = load_linnerud()
    X = d.data
    Y = d.target
    est = Est(n_components=1)
    one_d_coeff = est.fit(X, Y[:, 0]).coef_
    two_d_coeff = est.fit(X, Y[:, :1]).coef_
    assert one_d_coeff.shape == two_d_coeff.shape
    assert_array_almost_equal(one_d_coeff, two_d_coeff)