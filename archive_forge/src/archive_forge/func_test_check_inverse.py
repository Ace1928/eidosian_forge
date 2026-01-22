import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_check_inverse(sparse_container):
    X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))
    if sparse_container is not None:
        X = sparse_container(X)
    trans = FunctionTransformer(func=np.sqrt, inverse_func=np.around, accept_sparse=sparse_container is not None, check_inverse=True, validate=True)
    warning_message = "The provided functions are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'."
    with pytest.warns(UserWarning, match=warning_message):
        trans.fit(X)
    trans = FunctionTransformer(func=np.expm1, inverse_func=np.log1p, accept_sparse=sparse_container is not None, check_inverse=True, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        Xt = trans.fit_transform(X)
    assert_allclose_dense_sparse(X, trans.inverse_transform(Xt))