import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('pred_func', ['predict', 'predict_proba', 'decision_function', 'score'])
def test_check_X_on_predict_fail(iris, pred_func):
    X, y = iris
    clf = CheckingClassifier(check_X=_success).fit(X, y)
    clf.set_params(check_X=_fail)
    with pytest.raises(AssertionError):
        getattr(clf, pred_func)(X)