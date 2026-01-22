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
def test_feature_name_validation_missing_columns_drop_passthough():
    """Test the interaction between {'drop', 'passthrough'} and
    missing column names."""
    pd = pytest.importorskip('pandas')
    X = np.ones(shape=(3, 4))
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    df_dropped = df.drop('c', axis=1)
    tf = ColumnTransformer([('bycol', Trans(), [1])], remainder='passthrough')
    tf.fit(df)
    msg = "columns are missing: {'c'}"
    with pytest.raises(ValueError, match=msg):
        tf.transform(df_dropped)
    tf = ColumnTransformer([('bycol', Trans(), [1])], remainder='drop')
    tf.fit(df)
    df_dropped_trans = tf.transform(df_dropped)
    df_fit_trans = tf.transform(df)
    assert_allclose(df_dropped_trans, df_fit_trans)
    tf = ColumnTransformer([('bycol', 'drop', ['c'])], remainder='passthrough')
    tf.fit(df)
    df_dropped_trans = tf.transform(df_dropped)
    df_fit_trans = tf.transform(df)
    assert_allclose(df_dropped_trans, df_fit_trans)