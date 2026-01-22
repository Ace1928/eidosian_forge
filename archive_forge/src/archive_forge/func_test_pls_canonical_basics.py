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
def test_pls_canonical_basics():
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls = PLSCanonical(n_components=X.shape[1])
    pls.fit(X, Y)
    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)
    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)
    T = pls._x_scores
    P = pls.x_loadings_
    U = pls._y_scores
    Q = pls.y_loadings_
    Xc, Yc, x_mean, y_mean, x_std, y_std = _center_scale_xy(X.copy(), Y.copy(), scale=True)
    assert_array_almost_equal(Xc, np.dot(T, P.T))
    assert_array_almost_equal(Yc, np.dot(U, Q.T))
    Xt = pls.transform(X)
    assert_array_almost_equal(Xt, pls._x_scores)
    Xt, Yt = pls.transform(X, Y)
    assert_array_almost_equal(Xt, pls._x_scores)
    assert_array_almost_equal(Yt, pls._y_scores)
    X_back = pls.inverse_transform(Xt)
    assert_array_almost_equal(X_back, X)
    _, Y_back = pls.inverse_transform(Xt, Yt)
    assert_array_almost_equal(Y_back, Y)