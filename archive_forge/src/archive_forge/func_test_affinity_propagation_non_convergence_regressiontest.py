import warnings
import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_affinity_propagation_non_convergence_regressiontest(global_dtype):
    X = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1]], dtype=global_dtype)
    af = AffinityPropagation(affinity='euclidean', max_iter=2, random_state=34)
    msg = 'Affinity propagation did not converge, this model may return degenerate cluster centers and labels.'
    with pytest.warns(ConvergenceWarning, match=msg):
        af.fit(X)
    assert_array_equal(np.array([0, 0, 0]), af.labels_)