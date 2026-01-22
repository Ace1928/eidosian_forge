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
def test_affinity_propagation_no_copy():
    """Check behaviour of not copying the input data."""
    S = -euclidean_distances(X, squared=True)
    S_original = S.copy()
    preference = np.median(S) * 10
    assert not np.allclose(S.diagonal(), preference)
    affinity_propagation(S, preference=preference, copy=True, random_state=0)
    assert_allclose(S, S_original)
    assert not np.allclose(S.diagonal(), preference)
    assert_allclose(S.diagonal(), np.zeros(S.shape[0]))
    affinity_propagation(S, preference=preference, copy=False, random_state=0)
    assert_allclose(S.diagonal(), preference)
    S = S_original.copy()
    af = AffinityPropagation(preference=preference, verbose=True, random_state=0)
    labels = af.fit(X).labels_
    _, labels_no_copy = affinity_propagation(S, preference=preference, copy=False, random_state=74)
    assert_array_equal(labels, labels_no_copy)