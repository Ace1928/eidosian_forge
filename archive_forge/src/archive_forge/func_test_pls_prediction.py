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
@pytest.mark.parametrize('scale', [True, False])
@pytest.mark.parametrize('PLSEstimator', [PLSRegression, PLSCanonical, CCA])
def test_pls_prediction(PLSEstimator, scale):
    """Check the behaviour of the prediction function."""
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls = PLSEstimator(copy=True, scale=scale).fit(X, Y)
    Y_pred = pls.predict(X, copy=True)
    y_mean = Y.mean(axis=0)
    X_trans = X - X.mean(axis=0)
    if scale:
        X_trans /= X.std(axis=0, ddof=1)
    assert_allclose(pls.intercept_, y_mean)
    assert_allclose(Y_pred, X_trans @ pls.coef_.T + pls.intercept_)