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
def test_dataframe_different_dataframe_libraries():
    """Check fitting and transforming on pandas and polars dataframes."""
    pd = pytest.importorskip('pandas')
    pl = pytest.importorskip('polars')
    X_train_np = np.array([[0, 1], [2, 4], [4, 5]])
    X_test_np = np.array([[1, 2], [1, 3], [2, 3]])
    X_train_pd = pd.DataFrame(X_train_np, columns=['a', 'b'])
    X_test_pl = pl.DataFrame(X_test_np, schema=['a', 'b'])
    ct = make_column_transformer((Trans(), [0, 1]))
    ct.fit(X_train_pd)
    out_pl_in = ct.transform(X_test_pl)
    assert_array_equal(out_pl_in, X_test_np)
    X_train_pl = pl.DataFrame(X_train_np, schema=['a', 'b'])
    X_test_pd = pd.DataFrame(X_test_np, columns=['a', 'b'])
    ct.fit(X_train_pl)
    out_pd_in = ct.transform(X_test_pd)
    assert_array_equal(out_pd_in, X_test_np)