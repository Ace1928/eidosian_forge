import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_function_transformer_validate_inverse():
    """Test that function transformer does not reset estimator in
    `inverse_transform`."""

    def add_constant_feature(X):
        X_one = np.ones((X.shape[0], 1))
        return np.concatenate((X, X_one), axis=1)

    def inverse_add_constant(X):
        return X[:, :-1]
    X = np.array([[1, 2], [3, 4], [3, 4]])
    trans = FunctionTransformer(func=add_constant_feature, inverse_func=inverse_add_constant, validate=True)
    X_trans = trans.fit_transform(X)
    assert trans.n_features_in_ == X.shape[1]
    trans.inverse_transform(X_trans)
    assert trans.n_features_in_ == X.shape[1]