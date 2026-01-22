import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('Estimator', [svm.SVC, svm.SVR])
def test_custom_kernel_not_array_input(Estimator):
    """Test using a custom kernel that is not fed with array-like for floats"""
    data = ['A A', 'A', 'B', 'B B', 'A B']
    X = np.array([[2, 0], [1, 0], [0, 1], [0, 2], [1, 1]])
    y = np.array([1, 1, 2, 2, 1])

    def string_kernel(X1, X2):
        assert isinstance(X1[0], str)
        n_samples1 = _num_samples(X1)
        n_samples2 = _num_samples(X2)
        K = np.zeros((n_samples1, n_samples2))
        for ii in range(n_samples1):
            for jj in range(ii, n_samples2):
                K[ii, jj] = X1[ii].count('A') * X2[jj].count('A')
                K[ii, jj] += X1[ii].count('B') * X2[jj].count('B')
                K[jj, ii] = K[ii, jj]
        return K
    K = string_kernel(data, data)
    assert_array_equal(np.dot(X, X.T), K)
    svc1 = Estimator(kernel=string_kernel).fit(data, y)
    svc2 = Estimator(kernel='linear').fit(X, y)
    svc3 = Estimator(kernel='precomputed').fit(K, y)
    assert svc1.score(data, y) == svc3.score(K, y)
    assert svc1.score(data, y) == svc2.score(X, y)
    if hasattr(svc1, 'decision_function'):
        assert_allclose(svc1.decision_function(data), svc2.decision_function(X))
        assert_allclose(svc1.decision_function(data), svc3.decision_function(K))
        assert_array_equal(svc1.predict(data), svc2.predict(X))
        assert_array_equal(svc1.predict(data), svc3.predict(K))
    else:
        assert_allclose(svc1.predict(data), svc2.predict(X))
        assert_allclose(svc1.predict(data), svc3.predict(K))