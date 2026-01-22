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
def test_linearsvc_iris():
    target = iris.target_names[iris.target]
    clf = svm.LinearSVC(dual='auto', random_state=0).fit(iris.data, target)
    assert set(clf.classes_) == set(iris.target_names)
    assert np.mean(clf.predict(iris.data) == target) > 0.8
    dec = clf.decision_function(iris.data)
    pred = iris.target_names[np.argmax(dec, 1)]
    assert_array_equal(pred, clf.predict(iris.data))