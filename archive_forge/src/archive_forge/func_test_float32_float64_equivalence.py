import numpy as np
import pytest
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS + [None])
def test_float32_float64_equivalence(csr_container):
    """Check that the results are the same between float32 and float64."""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    if csr_container is not None:
        X[X < 0.8] = 0
        X = csr_container(X)
    km64 = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    km32 = BisectingKMeans(n_clusters=3, random_state=0).fit(X.astype(np.float32))
    assert_allclose(km32.cluster_centers_, km64.cluster_centers_)
    assert_array_equal(km32.labels_, km64.labels_)