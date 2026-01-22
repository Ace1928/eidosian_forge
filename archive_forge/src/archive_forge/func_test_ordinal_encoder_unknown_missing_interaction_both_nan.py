import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X_train, X_test_trans_expected, X_roundtrip_expected', [(np.array([['a'], ['1']], dtype=object), [[0], [np.nan], [np.nan]], np.asarray([['1'], [None], [None]], dtype=object)), (np.array([[np.nan], ['1'], ['a']], dtype=object), [[0], [np.nan], [np.nan]], np.asarray([['1'], [np.nan], [np.nan]], dtype=object))])
def test_ordinal_encoder_unknown_missing_interaction_both_nan(X_train, X_test_trans_expected, X_roundtrip_expected):
    """Check transform when unknown_value and encoded_missing_value is nan.

    Non-regression test for #24082.
    """
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan).fit(X_train)
    X_test = np.array([['1'], [np.nan], ['b']])
    X_test_trans = oe.transform(X_test)
    assert_allclose(X_test_trans, X_test_trans_expected)
    X_roundtrip = oe.inverse_transform(X_test_trans)
    n_samples = X_roundtrip_expected.shape[0]
    for i in range(n_samples):
        expected_val = X_roundtrip_expected[i, 0]
        val = X_roundtrip[i, 0]
        if expected_val is None:
            assert val is None
        elif is_scalar_nan(expected_val):
            assert np.isnan(val)
        else:
            assert val == expected_val