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
def test_affinity_propagation_predict_error():
    af = AffinityPropagation(affinity='euclidean')
    with pytest.raises(NotFittedError):
        af.predict(X)
    S = np.dot(X, X.T)
    af = AffinityPropagation(affinity='precomputed', random_state=57)
    af.fit(S)
    with pytest.raises(ValueError, match='expecting 60 features as input'):
        af.predict(X)