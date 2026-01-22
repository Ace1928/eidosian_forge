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
@pytest.mark.parametrize('Est', (CCA, PLSCanonical, PLSRegression, PLSSVD))
@pytest.mark.parametrize('X, Y', _generate_test_scale_and_stability_datasets())
def test_scale_and_stability(Est, X, Y):
    """scale=True is equivalent to scale=False on centered/scaled data
    This allows to check numerical stability over platforms as well"""
    X_s, Y_s, *_ = _center_scale_xy(X, Y)
    X_score, Y_score = Est(scale=True).fit_transform(X, Y)
    X_s_score, Y_s_score = Est(scale=False).fit_transform(X_s, Y_s)
    assert_allclose(X_s_score, X_score, atol=0.0001)
    assert_allclose(Y_s_score, Y_score, atol=0.0001)