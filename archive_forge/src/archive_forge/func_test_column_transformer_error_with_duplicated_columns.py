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
def test_column_transformer_error_with_duplicated_columns(dataframe_lib):
    """Check that we raise an error when using `ColumnTransformer` and
    the columns names are duplicated between transformers."""
    lib = pytest.importorskip(dataframe_lib)
    df = lib.DataFrame({'x1': [1, 2, 3], 'x2': [10, 20, 30], 'x3': [100, 200, 300]})
    transformer = ColumnTransformer(transformers=[('A', 'passthrough', ['x1', 'x2', 'x3']), ('B', FunctionTransformer(), ['x1', 'x2']), ('C', StandardScaler(), ['x1', 'x3']), ('D', FunctionTransformer(lambda x: x[[]]), ['x1', 'x2', 'x3'])], verbose_feature_names_out=False).set_output(transform=dataframe_lib)
    err_msg = re.escape("Duplicated feature names found before concatenating the outputs of the transformers: ['x1', 'x2', 'x3'].\nTransformer A has conflicting columns names: ['x1', 'x2', 'x3'].\nTransformer B has conflicting columns names: ['x1', 'x2'].\nTransformer C has conflicting columns names: ['x1', 'x3'].\n")
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit_transform(df)