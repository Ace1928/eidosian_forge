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
def test_libsvm_convergence_warnings():
    a = svm.SVC(kernel=lambda x, y: np.dot(x, y.T), probability=True, random_state=0, max_iter=2)
    warning_msg = 'Solver terminated early \\(max_iter=2\\).  Consider pre-processing your data with StandardScaler or MinMaxScaler.'
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        a.fit(np.array(X), Y)
    assert np.all(a.n_iter_ == 2)