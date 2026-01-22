import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{'max_categories': 1}, {'min_frequency': 100}])
def test_ordinal_encoder_all_infrequent(kwargs):
    """When all categories are infrequent, they are all encoded as zero."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3], dtype=object).T
    encoder = OrdinalEncoder(**kwargs, handle_unknown='use_encoded_value', unknown_value=-1).fit(X_train)
    X_test = [['a'], ['b'], ['c'], ['d'], ['e']]
    assert_allclose(encoder.transform(X_test), [[0], [0], [0], [0], [-1]])