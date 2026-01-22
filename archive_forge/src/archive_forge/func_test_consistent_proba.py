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
@ignore_warnings
def test_consistent_proba():
    a = svm.SVC(probability=True, max_iter=1, random_state=0)
    proba_1 = a.fit(X, Y).predict_proba(X)
    a = svm.SVC(probability=True, max_iter=1, random_state=0)
    proba_2 = a.fit(X, Y).predict_proba(X)
    assert_array_almost_equal(proba_1, proba_2)