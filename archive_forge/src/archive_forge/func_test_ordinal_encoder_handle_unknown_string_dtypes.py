import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X_train', [[['AA', 'B']], np.array([['AA', 'B']], dtype='O'), np.array([['AA', 'B']], dtype='U')])
@pytest.mark.parametrize('X_test', [[['A', 'B']], np.array([['A', 'B']], dtype='O'), np.array([['A', 'B']], dtype='U')])
def test_ordinal_encoder_handle_unknown_string_dtypes(X_train, X_test):
    """Checks that `OrdinalEncoder` transforms string dtypes.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19872
    """
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-9)
    enc.fit(X_train)
    X_trans = enc.transform(X_test)
    assert_allclose(X_trans, [[-9, 0]])