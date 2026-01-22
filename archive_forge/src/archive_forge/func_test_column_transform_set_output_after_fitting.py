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
def test_column_transform_set_output_after_fitting(remainder):
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'pet': pd.Series(['dog', 'cat', 'snake'], dtype='category'), 'age': [1.4, 2.1, 4.4], 'height': [20, 40, 10]})
    ct = ColumnTransformer([('color_encode', OneHotEncoder(sparse_output=False, dtype='int16'), ['pet']), ('age', StandardScaler(), ['age'])], remainder=remainder, verbose_feature_names_out=False)
    X_trans = ct.fit_transform(df)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.dtype == 'float64'
    ct.set_output(transform='pandas')
    X_trans_df = ct.transform(df)
    expected_dtypes = {'pet_cat': 'int16', 'pet_dog': 'int16', 'pet_snake': 'int16', 'height': 'int64', 'age': 'float64'}
    for col, dtype in X_trans_df.dtypes.items():
        assert dtype == expected_dtypes[col]