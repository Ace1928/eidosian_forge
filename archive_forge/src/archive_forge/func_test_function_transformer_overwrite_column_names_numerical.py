import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('feature_names_out', ['one-to-one', lambda _, names: [f'{name}_log' for name in names]])
def test_function_transformer_overwrite_column_names_numerical(feature_names_out):
    """Check the same as `test_function_transformer_overwrite_column_names`
    but for the specific case of pandas where column names can be numerical."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({0: [1, 2, 3], 1: [10, 20, 100]})
    transformer = FunctionTransformer(feature_names_out=feature_names_out)
    X_trans = transformer.fit_transform(df)
    assert_array_equal(np.asarray(X_trans), np.asarray(df))
    feature_names = transformer.get_feature_names_out()
    assert list(X_trans.columns) == list(feature_names)