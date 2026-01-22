import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_function_transformer_support_all_nummerical_dataframes_check_inverse_True():
    """Check support for dataframes with only numerical values."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    transformer = FunctionTransformer(func=lambda x: x + 2, inverse_func=lambda x: x - 2, check_inverse=True)
    df_out = transformer.fit_transform(df)
    assert_allclose_dense_sparse(df_out, df + 2)