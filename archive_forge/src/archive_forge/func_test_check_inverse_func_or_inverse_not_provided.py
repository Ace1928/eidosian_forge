import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_check_inverse_func_or_inverse_not_provided():
    X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))
    trans = FunctionTransformer(func=np.expm1, inverse_func=None, check_inverse=True, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        trans.fit(X)
    trans = FunctionTransformer(func=None, inverse_func=np.expm1, check_inverse=True, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        trans.fit(X)