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
@pytest.mark.parametrize('estimator, expected_n_iter_type', [(svm.SVC, np.ndarray), (svm.NuSVC, np.ndarray), (svm.SVR, int), (svm.NuSVR, int), (svm.OneClassSVM, int)])
@pytest.mark.parametrize('dataset', [make_classification(n_classes=2, n_informative=2, random_state=0), make_classification(n_classes=3, n_informative=3, random_state=0), make_classification(n_classes=4, n_informative=4, random_state=0)])
def test_n_iter_libsvm(estimator, expected_n_iter_type, dataset):
    X, y = dataset
    n_iter = estimator(kernel='linear').fit(X, y).n_iter_
    assert type(n_iter) == expected_n_iter_type
    if estimator in [svm.SVC, svm.NuSVC]:
        n_classes = len(np.unique(y))
        assert n_iter.shape == (n_classes * (n_classes - 1) // 2,)