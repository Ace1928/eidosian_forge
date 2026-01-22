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
@pytest.mark.parametrize('dataframe_lib', ['pandas', 'polars'])
def test_column_transformer_column_renaming(dataframe_lib):
    """Check that we properly rename columns when using `ColumnTransformer` and
    selected columns are redundant between transformers.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28260
    """
    lib = pytest.importorskip(dataframe_lib)
    df = lib.DataFrame({'x1': [1, 2, 3], 'x2': [10, 20, 30], 'x3': [100, 200, 300]})
    transformer = ColumnTransformer(transformers=[('A', 'passthrough', ['x1', 'x2', 'x3']), ('B', FunctionTransformer(), ['x1', 'x2']), ('C', StandardScaler(), ['x1', 'x3']), ('D', FunctionTransformer(lambda x: x[[]]), ['x1', 'x2', 'x3'])], verbose_feature_names_out=True).set_output(transform=dataframe_lib)
    df_trans = transformer.fit_transform(df)
    assert list(df_trans.columns) == ['A__x1', 'A__x2', 'A__x3', 'B__x1', 'B__x2', 'C__x1', 'C__x3']