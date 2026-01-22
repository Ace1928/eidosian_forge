import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_drop_idx_infrequent_categories():
    """Check drop_idx is defined correctly with infrequent categories.

    Non-regression test for gh-25550.
    """
    X = np.array([['a'] * 2 + ['b'] * 4 + ['c'] * 4 + ['d'] * 4 + ['e'] * 4], dtype=object).T
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop='first').fit(X)
    assert_array_equal(ohe.get_feature_names_out(), ['x0_c', 'x0_d', 'x0_e', 'x0_infrequent_sklearn'])
    assert ohe.categories_[0][ohe.drop_idx_[0]] == 'b'
    X = np.array([['a'] * 2 + ['b'] * 2 + ['c'] * 10], dtype=object).T
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop='if_binary').fit(X)
    assert_array_equal(ohe.get_feature_names_out(), ['x0_infrequent_sklearn'])
    assert ohe.categories_[0][ohe.drop_idx_[0]] == 'c'
    X = np.array([['a'] * 2 + ['b'] * 4 + ['c'] * 4 + ['d'] * 4 + ['e'] * 4], dtype=object).T
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=['d']).fit(X)
    assert_array_equal(ohe.get_feature_names_out(), ['x0_b', 'x0_c', 'x0_e', 'x0_infrequent_sklearn'])
    assert ohe.categories_[0][ohe.drop_idx_[0]] == 'd'
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=None).fit(X)
    assert_array_equal(ohe.get_feature_names_out(), ['x0_b', 'x0_c', 'x0_d', 'x0_e', 'x0_infrequent_sklearn'])
    assert ohe.drop_idx_ is None