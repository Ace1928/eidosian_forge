from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.manifold import _mds as mds
from sklearn.metrics import euclidean_distances
@pytest.mark.parametrize('metric', [True, False])
def test_normalized_stress_auto(metric, monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.randn(4, 3)
    dist = euclidean_distances(X)
    mock = Mock(side_effect=mds._smacof_single)
    monkeypatch.setattr('sklearn.manifold._mds._smacof_single', mock)
    est = mds.MDS(metric=metric, normalized_stress='auto', random_state=rng)
    est.fit_transform(X)
    assert mock.call_args[1]['normalized_stress'] != metric
    mds.smacof(dist, metric=metric, normalized_stress='auto', random_state=rng)
    assert mock.call_args[1]['normalized_stress'] != metric