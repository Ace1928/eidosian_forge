import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils._testing import assert_allclose, assert_no_warnings
def test_transform_target_regressor_invertible():
    X, y = friedman
    regr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.sqrt, inverse_func=np.log, check_inverse=True)
    with pytest.warns(UserWarning, match='The provided functions or transformer are not strictly inverse of each other.'):
        regr.fit(X, y)
    regr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.sqrt, inverse_func=np.log)
    regr.set_params(check_inverse=False)
    assert_no_warnings(regr.fit, X, y)