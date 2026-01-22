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
def test_iforest_average_path_length():
    result_one = 2.0 * (np.log(4.0) + np.euler_gamma) - 2.0 * 4.0 / 5.0
    result_two = 2.0 * (np.log(998.0) + np.euler_gamma) - 2.0 * 998.0 / 999.0
    assert_allclose(_average_path_length([0]), [0.0])
    assert_allclose(_average_path_length([1]), [0.0])
    assert_allclose(_average_path_length([2]), [1.0])
    assert_allclose(_average_path_length([5]), [result_one])
    assert_allclose(_average_path_length([999]), [result_two])
    assert_allclose(_average_path_length(np.array([1, 2, 5, 999])), [0.0, 1.0, result_one, result_two])
    avg_path_length = _average_path_length(np.arange(5))
    assert_array_equal(avg_path_length, np.sort(avg_path_length))