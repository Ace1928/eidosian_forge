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
def test_transform_target_regressor_error():
    X, y = friedman
    regr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler(), func=np.exp, inverse_func=np.log)
    with pytest.raises(ValueError, match="'transformer' and functions 'func'/'inverse_func' cannot both be set."):
        regr.fit(X, y)
    sample_weight = np.ones((y.shape[0],))
    regr = TransformedTargetRegressor(regressor=OrthogonalMatchingPursuit(), transformer=StandardScaler())
    with pytest.raises(TypeError, match="fit\\(\\) got an unexpected keyword argument 'sample_weight'"):
        regr.fit(X, y, sample_weight=sample_weight)
    regr = TransformedTargetRegressor(func=np.exp)
    with pytest.raises(ValueError, match="When 'func' is provided, 'inverse_func' must also be provided"):
        regr.fit(X, y)