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
@pytest.mark.parametrize('estimator', [svm.SVC(C=0.01), svm.NuSVC()])
def test_svm_classifier_sided_sample_weight(estimator):
    X = [[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 0]]
    estimator.set_params(kernel='linear')
    sample_weight = [1] * 6
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred == pytest.approx(0)
    sample_weight = [10.0, 0.1, 0.1, 0.1, 0.1, 10]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred < 0
    sample_weight = [1.0, 0.1, 10.0, 10.0, 0.1, 0.1]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred > 0