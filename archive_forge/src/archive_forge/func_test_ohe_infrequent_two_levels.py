import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{'max_categories': 2}, {'min_frequency': 11}, {'min_frequency': 0.29}, {'max_categories': 2, 'min_frequency': 6}, {'max_categories': 4, 'min_frequency': 12}])
@pytest.mark.parametrize('categories', ['auto', [['a', 'b', 'c', 'd']]])
def test_ohe_infrequent_two_levels(kwargs, categories):
    """Test that different parameters for combine 'a', 'c', and 'd' into
    the infrequent category works as expected."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ohe = OneHotEncoder(categories=categories, handle_unknown='infrequent_if_exist', sparse_output=False, **kwargs).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [['a', 'c', 'd']])
    X_test = [['b'], ['a'], ['c'], ['d'], ['e']]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)
    expected_inv = [[col] for col in ['b'] + ['infrequent_sklearn'] * 4]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)
    feature_names = ohe.get_feature_names_out()
    assert_array_equal(['x0_b', 'x0_infrequent_sklearn'], feature_names)