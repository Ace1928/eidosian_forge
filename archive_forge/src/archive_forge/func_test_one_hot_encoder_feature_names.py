import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_feature_names():
    enc = OneHotEncoder()
    X = [['Male', 1, 'girl', 2, 3], ['Female', 41, 'girl', 1, 10], ['Male', 51, 'boy', 12, 3], ['Male', 91, 'girl', 21, 30]]
    enc.fit(X)
    feature_names = enc.get_feature_names_out()
    assert_array_equal(['x0_Female', 'x0_Male', 'x1_1', 'x1_41', 'x1_51', 'x1_91', 'x2_boy', 'x2_girl', 'x3_1', 'x3_2', 'x3_12', 'x3_21', 'x4_3', 'x4_10', 'x4_30'], feature_names)
    feature_names2 = enc.get_feature_names_out(['one', 'two', 'three', 'four', 'five'])
    assert_array_equal(['one_Female', 'one_Male', 'two_1', 'two_41', 'two_51', 'two_91', 'three_boy', 'three_girl', 'four_1', 'four_2', 'four_12', 'four_21', 'five_3', 'five_10', 'five_30'], feature_names2)
    with pytest.raises(ValueError, match='input_features should have length'):
        enc.get_feature_names_out(['one', 'two'])