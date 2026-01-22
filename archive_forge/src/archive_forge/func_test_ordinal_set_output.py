import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_set_output():
    """Check OrdinalEncoder works with set_output."""
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'A': ['a', 'b'], 'B': [1, 2]})
    ord_default = OrdinalEncoder().set_output(transform='default')
    ord_pandas = OrdinalEncoder().set_output(transform='pandas')
    X_default = ord_default.fit_transform(X_df)
    X_pandas = ord_pandas.fit_transform(X_df)
    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(ord_pandas.get_feature_names_out(), X_pandas.columns)