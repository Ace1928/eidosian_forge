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
def test_libsvm_iris():
    for k in ('linear', 'rbf'):
        clf = svm.SVC(kernel=k).fit(iris.data, iris.target)
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.9
        assert hasattr(clf, 'coef_') == (k == 'linear')
    assert_array_equal(clf.classes_, np.sort(clf.classes_))
    libsvm_support, libsvm_support_vectors, libsvm_n_class_SV, libsvm_sv_coef, libsvm_intercept, libsvm_probA, libsvm_probB, libsvm_fit_status, libsvm_n_iter = _libsvm.fit(iris.data, iris.target.astype(np.float64))
    model_params = {'support': libsvm_support, 'SV': libsvm_support_vectors, 'nSV': libsvm_n_class_SV, 'sv_coef': libsvm_sv_coef, 'intercept': libsvm_intercept, 'probA': libsvm_probA, 'probB': libsvm_probB}
    pred = _libsvm.predict(iris.data, **model_params)
    assert np.mean(pred == iris.target) > 0.95
    libsvm_support, libsvm_support_vectors, libsvm_n_class_SV, libsvm_sv_coef, libsvm_intercept, libsvm_probA, libsvm_probB, libsvm_fit_status, libsvm_n_iter = _libsvm.fit(iris.data, iris.target.astype(np.float64), kernel='linear')
    model_params = {'support': libsvm_support, 'SV': libsvm_support_vectors, 'nSV': libsvm_n_class_SV, 'sv_coef': libsvm_sv_coef, 'intercept': libsvm_intercept, 'probA': libsvm_probA, 'probB': libsvm_probB}
    pred = _libsvm.predict(iris.data, **model_params, kernel='linear')
    assert np.mean(pred == iris.target) > 0.95
    pred = _libsvm.cross_validation(iris.data, iris.target.astype(np.float64), 5, kernel='linear', random_seed=0)
    assert np.mean(pred == iris.target) > 0.95
    pred2 = _libsvm.cross_validation(iris.data, iris.target.astype(np.float64), 5, kernel='linear', random_seed=0)
    assert_array_equal(pred, pred2)