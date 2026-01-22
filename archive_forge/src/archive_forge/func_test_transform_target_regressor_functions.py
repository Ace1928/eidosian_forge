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
def test_transform_target_regressor_functions():
    X, y = friedman
    regr = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)
    y_pred = regr.fit(X, y).predict(X)
    y_tran = regr.transformer_.transform(y.reshape(-1, 1)).squeeze()
    assert_allclose(np.log(y), y_tran)
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran.reshape(-1, 1)).squeeze())
    assert y.shape == y_pred.shape
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))
    lr = LinearRegression().fit(X, regr.func(y))
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())