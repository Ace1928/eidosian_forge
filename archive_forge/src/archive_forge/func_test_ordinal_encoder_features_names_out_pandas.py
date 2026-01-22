import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_features_names_out_pandas():
    """Check feature names out is same as the input."""
    pd = pytest.importorskip('pandas')
    names = ['b', 'c', 'a']
    X = pd.DataFrame([[1, 2, 3]], columns=names)
    enc = OrdinalEncoder().fit(X)
    feature_names_out = enc.get_feature_names_out()
    assert_array_equal(names, feature_names_out)