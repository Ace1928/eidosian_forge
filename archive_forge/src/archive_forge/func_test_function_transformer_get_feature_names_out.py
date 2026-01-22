import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('X, feature_names_out, input_features, expected', [(np.random.rand(100, 3), 'one-to-one', None, ('x0', 'x1', 'x2')), ({'a': np.random.rand(100), 'b': np.random.rand(100)}, 'one-to-one', None, ('a', 'b')), (np.random.rand(100, 3), lambda transformer, input_features: ('a', 'b'), None, ('a', 'b')), ({'a': np.random.rand(100), 'b': np.random.rand(100)}, lambda transformer, input_features: ('c', 'd', 'e'), None, ('c', 'd', 'e')), (np.random.rand(100, 3), lambda transformer, input_features: tuple(input_features) + ('a',), None, ('x0', 'x1', 'x2', 'a')), ({'a': np.random.rand(100), 'b': np.random.rand(100)}, lambda transformer, input_features: tuple(input_features) + ('c',), None, ('a', 'b', 'c')), (np.random.rand(100, 3), 'one-to-one', ('a', 'b', 'c'), ('a', 'b', 'c')), ({'a': np.random.rand(100), 'b': np.random.rand(100)}, 'one-to-one', ('a', 'b'), ('a', 'b')), (np.random.rand(100, 3), lambda transformer, input_features: tuple(input_features) + ('d',), ('a', 'b', 'c'), ('a', 'b', 'c', 'd')), ({'a': np.random.rand(100), 'b': np.random.rand(100)}, lambda transformer, input_features: tuple(input_features) + ('c',), ('a', 'b'), ('a', 'b', 'c'))])
@pytest.mark.parametrize('validate', [True, False])
def test_function_transformer_get_feature_names_out(X, feature_names_out, input_features, expected, validate):
    if isinstance(X, dict):
        pd = pytest.importorskip('pandas')
        X = pd.DataFrame(X)
    transformer = FunctionTransformer(feature_names_out=feature_names_out, validate=validate)
    transformer.fit(X)
    names = transformer.get_feature_names_out(input_features)
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected)