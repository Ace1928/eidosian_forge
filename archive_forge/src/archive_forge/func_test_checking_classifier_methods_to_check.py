import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('methods_to_check', [['predict'], ['predict', 'predict_proba']])
@pytest.mark.parametrize('predict_method', ['predict', 'predict_proba', 'decision_function', 'score'])
def test_checking_classifier_methods_to_check(iris, methods_to_check, predict_method):
    X, y = iris
    clf = CheckingClassifier(check_X=sparse.issparse, methods_to_check=methods_to_check)
    clf.fit(X, y)
    if predict_method in methods_to_check:
        with pytest.raises(AssertionError):
            getattr(clf, predict_method)(X)
    else:
        getattr(clf, predict_method)(X)