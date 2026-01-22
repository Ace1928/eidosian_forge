import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_consistence_column_name_between_steps():
    """Check that we have a consistence between the feature names out of
    `FunctionTransformer` and the feature names in of the next step in the pipeline.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27695
    """
    pd = pytest.importorskip('pandas')

    def with_suffix(_, names):
        return [name + '__log' for name in names]
    pipeline = make_pipeline(FunctionTransformer(np.log1p, feature_names_out=with_suffix), StandardScaler())
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['a', 'b'])
    X_trans = pipeline.fit_transform(df)
    assert pipeline.get_feature_names_out().tolist() == ['a__log', 'b__log']
    assert isinstance(X_trans, np.ndarray)