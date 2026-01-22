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
@pytest.mark.parametrize('Klass', (OneClassSVM, SVR, NuSVR))
def test_n_support(Klass):
    X = np.array([[0], [0.44], [0.45], [0.46], [1]])
    y = np.arange(X.shape[0])
    est = Klass()
    assert not hasattr(est, 'n_support_')
    est.fit(X, y)
    assert est.n_support_[0] == est.support_vectors_.shape[0]
    assert est.n_support_.size == 1