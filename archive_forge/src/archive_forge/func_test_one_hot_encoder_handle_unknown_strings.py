import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('handle_unknown', ['ignore', 'infrequent_if_exist'])
def test_one_hot_encoder_handle_unknown_strings(handle_unknown):
    X = np.array(['11111111', '22', '333', '4444']).reshape((-1, 1))
    X2 = np.array(['55555', '22']).reshape((-1, 1))
    oh = OneHotEncoder(handle_unknown=handle_unknown)
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(oh.transform(X2_passed).toarray(), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
    assert_array_equal(X2, X2_passed)