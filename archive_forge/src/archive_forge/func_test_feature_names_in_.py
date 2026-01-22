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
def test_feature_names_in_():
    """Feature names are stored in column transformer.

    Column transformer deliberately does not check for column name consistency.
    It only checks that the non-dropped names seen in `fit` are seen
    in `transform`. This behavior is already tested in
    `test_feature_name_validation_missing_columns_drop_passthough`"""
    pd = pytest.importorskip('pandas')
    feature_names = ['a', 'c', 'd']
    df = pd.DataFrame([[1, 2, 3]], columns=feature_names)
    ct = ColumnTransformer([('bycol', Trans(), ['a', 'd'])], remainder='passthrough')
    ct.fit(df)
    assert_array_equal(ct.feature_names_in_, feature_names)
    assert isinstance(ct.feature_names_in_, np.ndarray)
    assert ct.feature_names_in_.dtype == object