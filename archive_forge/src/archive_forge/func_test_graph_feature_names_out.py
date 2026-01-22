import numpy as np
import pytest
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.neighbors._base import _is_sorted_by_data
from sklearn.utils._testing import assert_array_equal
@pytest.mark.parametrize('Klass', [KNeighborsTransformer, RadiusNeighborsTransformer])
def test_graph_feature_names_out(Klass):
    """Check `get_feature_names_out` for transformers defined in `_graph.py`."""
    n_samples_fit = 20
    n_features = 10
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    est = Klass().fit(X)
    names_out = est.get_feature_names_out()
    class_name_lower = Klass.__name__.lower()
    expected_names_out = np.array([f'{class_name_lower}{i}' for i in range(est.n_samples_fit_)], dtype=object)
    assert_array_equal(names_out, expected_names_out)