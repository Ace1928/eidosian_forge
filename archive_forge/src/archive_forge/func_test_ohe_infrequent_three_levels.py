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
def test_ohe_infrequent_three_levels(kwargs):
    """Test that different parameters for combing 'a', and 'd' into
    the infrequent category works as expected."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, **kwargs).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [['a', 'd']])
    X_test = [['b'], ['a'], ['c'], ['d'], ['e']]
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)
    expected_inv = [['b'], ['infrequent_sklearn'], ['c'], ['infrequent_sklearn'], ['infrequent_sklearn']]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)
    feature_names = ohe.get_feature_names_out()
    assert_array_equal(['x0_b', 'x0_c', 'x0_infrequent_sklearn'], feature_names)