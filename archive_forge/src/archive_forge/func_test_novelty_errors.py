import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_novelty_errors():
    X = iris.data
    clf = neighbors.LocalOutlierFactor()
    clf.fit(X)
    for method in ['predict', 'decision_function', 'score_samples']:
        outer_msg = f"'LocalOutlierFactor' has no attribute '{method}'"
        inner_msg = '{} is not available when novelty=False'.format(method)
        with pytest.raises(AttributeError, match=outer_msg) as exec_info:
            getattr(clf, method)
        assert isinstance(exec_info.value.__cause__, AttributeError)
        assert inner_msg in str(exec_info.value.__cause__)
    clf = neighbors.LocalOutlierFactor(novelty=True)
    outer_msg = "'LocalOutlierFactor' has no attribute 'fit_predict'"
    inner_msg = 'fit_predict is not available when novelty=True'
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        getattr(clf, 'fit_predict')
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)