import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{'max_categories': 3}, {'min_frequency': 6}, {'min_frequency': 9}, {'min_frequency': 0.24}, {'min_frequency': 0.16}, {'max_categories': 3, 'min_frequency': 8}, {'max_categories': 4, 'min_frequency': 6}])
def test_ordinal_encoder_infrequent_three_levels(kwargs):
    """Test parameters for grouping 'a', and 'd' into the infrequent category."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, **kwargs).fit(X_train)
    assert_array_equal(ordinal.categories_, [['a', 'b', 'c', 'd']])
    assert_array_equal(ordinal.infrequent_categories_, [['a', 'd']])
    X_test = [['a'], ['b'], ['c'], ['d'], ['z']]
    expected_trans = [[2], [0], [1], [2], [-1]]
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)
    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = [['infrequent_sklearn'], ['b'], ['c'], ['infrequent_sklearn'], [None]]
    assert_array_equal(X_inverse, expected_inverse)