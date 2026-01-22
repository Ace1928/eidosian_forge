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
def test_iforest_with_uniform_data():
    """Test whether iforest predicts inliers when using uniform data"""
    X = np.ones((100, 10))
    iforest = IsolationForest()
    iforest.fit(X)
    rng = np.random.RandomState(0)
    assert all(iforest.predict(X) == 1)
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    assert all(iforest.predict(X + 1) == 1)
    assert all(iforest.predict(X - 1) == 1)
    X = np.repeat(rng.randn(1, 10), 100, 0)
    iforest = IsolationForest()
    iforest.fit(X)
    assert all(iforest.predict(X) == 1)
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    assert all(iforest.predict(np.ones((100, 10))) == 1)
    X = rng.randn(1, 10)
    iforest = IsolationForest()
    iforest.fit(X)
    assert all(iforest.predict(X) == 1)
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    assert all(iforest.predict(np.ones((100, 10))) == 1)