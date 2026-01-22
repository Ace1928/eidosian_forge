import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('encode, expected_names', [('onehot', [f'feat{col_id}_{float(bin_id)}' for col_id in range(3) for bin_id in range(4)]), ('onehot-dense', [f'feat{col_id}_{float(bin_id)}' for col_id in range(3) for bin_id in range(4)]), ('ordinal', [f'feat{col_id}' for col_id in range(3)])])
def test_kbinsdiscrtizer_get_feature_names_out(encode, expected_names):
    """Check get_feature_names_out for different settings.
    Non-regression test for #22731
    """
    X = [[-2, 1, -4], [-1, 2, -3], [0, 3, -2], [1, 4, -1]]
    kbd = KBinsDiscretizer(n_bins=4, encode=encode).fit(X)
    Xt = kbd.transform(X)
    input_features = [f'feat{i}' for i in range(3)]
    output_names = kbd.get_feature_names_out(input_features)
    assert Xt.shape[1] == output_names.shape[0]
    assert_array_equal(output_names, expected_names)