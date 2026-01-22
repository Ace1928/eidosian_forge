import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_fit_with_unseen_category():
    """Check OrdinalEncoder.fit works with unseen category when
    `handle_unknown="use_encoded_value"`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19872
    """
    X = np.array([0, 0, 1, 0, 2, 5])[:, np.newaxis]
    oe = OrdinalEncoder(categories=[[-1, 0, 1]], handle_unknown='use_encoded_value', unknown_value=-999)
    oe.fit(X)
    oe = OrdinalEncoder(categories=[[-1, 0, 1]], handle_unknown='error')
    with pytest.raises(ValueError, match='Found unknown categories'):
        oe.fit(X)