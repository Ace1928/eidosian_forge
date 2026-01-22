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
def test_iforest_error():
    """Test that it gives proper exception on deficient input."""
    X = iris.data
    warn_msg = 'max_samples will be set to n_samples for estimation'
    with pytest.warns(UserWarning, match=warn_msg):
        IsolationForest(max_samples=1000).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        IsolationForest(max_samples='auto').fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        IsolationForest(max_samples=np.int64(2)).fit(X)
    with pytest.raises(ValueError):
        IsolationForest().fit(X).predict(X[:, 1:])