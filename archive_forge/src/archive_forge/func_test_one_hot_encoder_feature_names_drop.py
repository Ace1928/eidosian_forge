import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('drop, expected_names', [('first', ['x0_c', 'x2_b']), ('if_binary', ['x0_c', 'x1_2', 'x2_b']), (['c', 2, 'b'], ['x0_b', 'x2_a'])], ids=['first', 'binary', 'manual'])
def test_one_hot_encoder_feature_names_drop(drop, expected_names):
    X = [['c', 2, 'a'], ['b', 2, 'b']]
    ohe = OneHotEncoder(drop=drop)
    ohe.fit(X)
    feature_names = ohe.get_feature_names_out()
    assert_array_equal(expected_names, feature_names)