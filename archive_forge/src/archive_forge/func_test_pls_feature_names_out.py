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
@pytest.mark.parametrize('Klass', [CCA, PLSSVD, PLSRegression, PLSCanonical])
def test_pls_feature_names_out(Klass):
    """Check `get_feature_names_out` cross_decomposition module."""
    X, Y = load_linnerud(return_X_y=True)
    est = Klass().fit(X, Y)
    names_out = est.get_feature_names_out()
    class_name_lower = Klass.__name__.lower()
    expected_names_out = np.array([f'{class_name_lower}{i}' for i in range(est.x_weights_.shape[1])], dtype=object)
    assert_array_equal(names_out, expected_names_out)