import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_function_transformer_frame():
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame(np.random.randn(100, 10))
    transformer = FunctionTransformer()
    X_df_trans = transformer.fit_transform(X_df)
    assert hasattr(X_df_trans, 'loc')