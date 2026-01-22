import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_set_output():
    """Check OneHotEncoder works with set_output."""
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'A': ['a', 'b'], 'B': [1, 2]})
    ohe = OneHotEncoder()
    ohe.set_output(transform='pandas')
    match = 'Pandas output does not support sparse data. Set sparse_output=False'
    with pytest.raises(ValueError, match=match):
        ohe.fit_transform(X_df)
    ohe_default = OneHotEncoder(sparse_output=False).set_output(transform='default')
    ohe_pandas = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    X_default = ohe_default.fit_transform(X_df)
    X_pandas = ohe_pandas.fit_transform(X_df)
    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(ohe_pandas.get_feature_names_out(), X_pandas.columns)