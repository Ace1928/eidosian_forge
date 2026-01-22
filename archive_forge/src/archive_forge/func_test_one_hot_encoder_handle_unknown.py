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
def test_one_hot_encoder_handle_unknown(handle_unknown):
    X = np.array([[0, 2, 1], [1, 0, 3], [1, 0, 2]])
    X2 = np.array([[4, 1, 1]])
    oh = OneHotEncoder(handle_unknown='error')
    oh.fit(X)
    with pytest.raises(ValueError, match='Found unknown categories'):
        oh.transform(X2)
    oh = OneHotEncoder(handle_unknown=handle_unknown)
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(oh.transform(X2_passed).toarray(), np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    assert_allclose(X2, X2_passed)