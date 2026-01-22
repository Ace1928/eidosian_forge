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
@pytest.mark.parametrize('verbose_feature_names_out', [True, False])
@pytest.mark.parametrize('remainder', ['drop', 'passthrough'])
def test_column_transformer_set_output(verbose_feature_names_out, remainder):
    """Check column transformer behavior with set_output."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'], index=[10])
    ct = ColumnTransformer([('first', TransWithNames(), ['a', 'c']), ('second', TransWithNames(), ['d'])], remainder=remainder, verbose_feature_names_out=verbose_feature_names_out)
    X_trans = ct.fit_transform(df)
    assert isinstance(X_trans, np.ndarray)
    ct.set_output(transform='pandas')
    df_test = pd.DataFrame([[1, 2, 3, 4]], columns=df.columns, index=[20])
    X_trans = ct.transform(df_test)
    assert isinstance(X_trans, pd.DataFrame)
    feature_names_out = ct.get_feature_names_out()
    assert_array_equal(X_trans.columns, feature_names_out)
    assert_array_equal(X_trans.index, df_test.index)