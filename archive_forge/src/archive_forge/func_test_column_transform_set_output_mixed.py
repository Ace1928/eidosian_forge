import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('remainder', ['drop', 'passthrough'])
@pytest.mark.parametrize('fit_transform', [True, False])
def test_column_transform_set_output_mixed(remainder, fit_transform):
    """Check ColumnTransformer outputs mixed types correctly."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'pet': pd.Series(['dog', 'cat', 'snake'], dtype='category'), 'color': pd.Series(['green', 'blue', 'red'], dtype='object'), 'age': [1.4, 2.1, 4.4], 'height': [20, 40, 10], 'distance': pd.Series([20, pd.NA, 100], dtype='Int32')})
    ct = ColumnTransformer([('color_encode', OneHotEncoder(sparse_output=False, dtype='int8'), ['color']), ('age', StandardScaler(), ['age'])], remainder=remainder, verbose_feature_names_out=False).set_output(transform='pandas')
    if fit_transform:
        X_trans = ct.fit_transform(df)
    else:
        X_trans = ct.fit(df).transform(df)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ct.get_feature_names_out())
    expected_dtypes = {'color_blue': 'int8', 'color_green': 'int8', 'color_red': 'int8', 'age': 'float64', 'pet': 'category', 'height': 'int64', 'distance': 'Int32'}
    for col, dtype in X_trans.dtypes.items():
        assert dtype == expected_dtypes[col]