import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_unknown_missing_interaction():
    """Check interactions between encode_unknown and missing value encoding."""
    X = np.array([['a'], ['b'], [np.nan]], dtype=object)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=-3).fit(X)
    X_trans = oe.transform(X)
    assert_allclose(X_trans, [[0], [1], [-3]])
    X_test = np.array([['c'], [np.nan]], dtype=object)
    X_test_trans = oe.transform(X_test)
    assert_allclose(X_test_trans, [[np.nan], [-3]])
    X_roundtrip = oe.inverse_transform(X_test_trans)
    assert X_roundtrip[0][0] is None
    assert np.isnan(X_roundtrip[1][0])