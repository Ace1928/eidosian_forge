import warnings
from unittest.mock import Mock, patch
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_iforest_sparse_input_float_contamination(sparse_container):
    """Check that `IsolationForest` accepts sparse matrix input and float value for
    contamination.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27626
    """
    X, _ = make_classification(n_samples=50, n_features=4, random_state=0)
    X = sparse_container(X)
    X.sort_indices()
    contamination = 0.1
    iforest = IsolationForest(n_estimators=5, contamination=contamination, random_state=0).fit(X)
    X_decision = iforest.decision_function(X)
    assert (X_decision < 0).sum() / X.shape[0] == pytest.approx(contamination)