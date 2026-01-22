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
def test_linear_svm_convergence_warnings():
    lsvc = svm.LinearSVC(dual='auto', random_state=0, max_iter=2)
    warning_msg = 'Liblinear failed to converge, increase the number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        lsvc.fit(X, Y)
    assert isinstance(lsvc.n_iter_, int)
    assert lsvc.n_iter_ == 2
    lsvr = svm.LinearSVR(dual='auto', random_state=0, max_iter=2)
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        lsvr.fit(iris.data, iris.target)
    assert isinstance(lsvr.n_iter_, int)
    assert lsvr.n_iter_ == 2