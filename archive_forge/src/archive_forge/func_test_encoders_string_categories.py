import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('input_dtype, category_dtype', ['OO', 'OU', 'UO', 'UU', 'SO', 'SU', 'SS'])
@pytest.mark.parametrize('array_type', ['list', 'array', 'dataframe'])
def test_encoders_string_categories(input_dtype, category_dtype, array_type):
    """Check that encoding work with object, unicode, and byte string dtypes.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15616
    https://github.com/scikit-learn/scikit-learn/issues/15726
    https://github.com/scikit-learn/scikit-learn/issues/19677
    """
    X = np.array([['b'], ['a']], dtype=input_dtype)
    categories = [np.array(['b', 'a'], dtype=category_dtype)]
    ohe = OneHotEncoder(categories=categories, sparse_output=False).fit(X)
    X_test = _convert_container([['a'], ['a'], ['b'], ['a']], array_type, dtype=input_dtype)
    X_trans = ohe.transform(X_test)
    expected = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])
    assert_allclose(X_trans, expected)
    oe = OrdinalEncoder(categories=categories).fit(X)
    X_trans = oe.transform(X_test)
    expected = np.array([[1], [1], [0], [1]])
    assert_array_equal(X_trans, expected)