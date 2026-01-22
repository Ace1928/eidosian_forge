import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_missing_unknown_encoding_max():
    """Check missing value or unknown encoding can equal the cardinality."""
    X = np.array([['dog'], ['cat'], [np.nan]], dtype=object)
    X_trans = OrdinalEncoder(encoded_missing_value=2).fit_transform(X)
    assert_allclose(X_trans, [[1], [0], [2]])
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=2).fit(X)
    X_test = np.array([['snake']])
    X_trans = enc.transform(X_test)
    assert_allclose(X_trans, [[2]])