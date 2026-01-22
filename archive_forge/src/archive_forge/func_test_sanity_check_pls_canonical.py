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
def test_sanity_check_pls_canonical():
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls = PLSCanonical(n_components=X.shape[1])
    pls.fit(X, Y)
    expected_x_weights = np.array([[-0.61330704, 0.25616119, -0.74715187], [-0.74697144, 0.11930791, 0.65406368], [-0.25668686, -0.95924297, -0.11817271]])
    expected_x_rotations = np.array([[-0.61330704, 0.41591889, -0.62297525], [-0.74697144, 0.31388326, 0.77368233], [-0.25668686, -0.89237972, -0.24121788]])
    expected_y_weights = np.array([[+0.58989127, 0.7890047, 0.1717553], [+0.77134053, -0.61351791, 0.16920272], [-0.2388767, -0.03267062, 0.97050016]])
    expected_y_rotations = np.array([[+0.58989127, 0.7168115, 0.30665872], [+0.77134053, -0.70791757, 0.19786539], [-0.2388767, -0.00343595, 0.94162826]])
    assert_array_almost_equal(np.abs(pls.x_rotations_), np.abs(expected_x_rotations))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_rotations_), np.abs(expected_y_rotations))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))
    x_rotations_sign_flip = np.sign(pls.x_rotations_ / expected_x_rotations)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_rotations_sign_flip = np.sign(pls.y_rotations_ / expected_y_rotations)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    assert_array_almost_equal(x_rotations_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_rotations_sign_flip, y_weights_sign_flip)
    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)
    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)