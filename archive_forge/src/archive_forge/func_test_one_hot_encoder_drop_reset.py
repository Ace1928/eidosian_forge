import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('drop', ['if_binary', 'first', None])
@pytest.mark.parametrize('reset_drop', ['if_binary', 'first', None])
def test_one_hot_encoder_drop_reset(drop, reset_drop):
    X = np.array([['Male', 1], ['Female', 3], ['Female', 2]], dtype=object)
    ohe = OneHotEncoder(drop=drop, sparse_output=False)
    ohe.fit(X)
    X_tr = ohe.transform(X)
    feature_names = ohe.get_feature_names_out()
    ohe.set_params(drop=reset_drop)
    assert_array_equal(ohe.inverse_transform(X_tr), X)
    assert_allclose(ohe.transform(X), X_tr)
    assert_array_equal(ohe.get_feature_names_out(), feature_names)