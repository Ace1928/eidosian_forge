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
def test_hasattr_predict_proba():
    G = svm.SVC(probability=True)
    assert hasattr(G, 'predict_proba')
    G.fit(iris.data, iris.target)
    assert hasattr(G, 'predict_proba')
    G = svm.SVC(probability=False)
    assert not hasattr(G, 'predict_proba')
    G.fit(iris.data, iris.target)
    assert not hasattr(G, 'predict_proba')
    G.probability = True
    assert hasattr(G, 'predict_proba')
    msg = 'predict_proba is not available when fitted with probability=False'
    with pytest.raises(NotFittedError, match=msg):
        G.predict_proba(iris.data)