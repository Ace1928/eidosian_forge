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
def test_transform_target_regressor_multi_to_single():
    X = friedman[0]
    y = np.transpose([friedman[1], friedman[1] ** 2 + 1])

    def func(y):
        out = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        return out[:, np.newaxis]

    def inverse_func(y):
        return y
    tt = TransformedTargetRegressor(func=func, inverse_func=inverse_func, check_inverse=False)
    tt.fit(X, y)
    y_pred_2d_func = tt.predict(X)
    assert y_pred_2d_func.shape == (100, 1)

    def func(y):
        return np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
    tt = TransformedTargetRegressor(func=func, inverse_func=inverse_func, check_inverse=False)
    tt.fit(X, y)
    y_pred_1d_func = tt.predict(X)
    assert y_pred_1d_func.shape == (100, 1)
    assert_allclose(y_pred_1d_func, y_pred_2d_func)