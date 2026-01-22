import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('dataframe_lib', ['pandas', 'polars'])
@pytest.mark.parametrize('transform_output', ['default', 'pandas', 'polars'])
def test_function_transformer_overwrite_column_names(dataframe_lib, transform_output):
    """Check that we overwrite the column names when we should."""
    lib = pytest.importorskip(dataframe_lib)
    if transform_output != 'numpy':
        pytest.importorskip(transform_output)
    df = lib.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 100]})

    def with_suffix(_, names):
        return [name + '__log' for name in names]
    transformer = FunctionTransformer(feature_names_out=with_suffix).set_output(transform=transform_output)
    X_trans = transformer.fit_transform(df)
    assert_array_equal(np.asarray(X_trans), np.asarray(df))
    feature_names = transformer.get_feature_names_out()
    assert list(X_trans.columns) == with_suffix(None, df.columns)
    assert feature_names.tolist() == with_suffix(None, df.columns)