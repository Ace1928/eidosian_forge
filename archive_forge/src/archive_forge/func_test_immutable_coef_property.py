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
def test_immutable_coef_property():
    svms = [svm.SVC(kernel='linear').fit(iris.data, iris.target), svm.NuSVC(kernel='linear').fit(iris.data, iris.target), svm.SVR(kernel='linear').fit(iris.data, iris.target), svm.NuSVR(kernel='linear').fit(iris.data, iris.target), svm.OneClassSVM(kernel='linear').fit(iris.data)]
    for clf in svms:
        with pytest.raises(AttributeError):
            clf.__setattr__('coef_', np.arange(3))
        with pytest.raises((RuntimeError, ValueError)):
            clf.coef_.__setitem__((0, 0), 0)