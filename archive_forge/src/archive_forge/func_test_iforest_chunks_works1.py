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
@patch('sklearn.ensemble._iforest.get_chunk_n_rows', side_effect=Mock(**{'return_value': 3}))
@pytest.mark.parametrize('contamination, n_predict_calls', [(0.25, 3), ('auto', 2)])
def test_iforest_chunks_works1(mocked_get_chunk, contamination, n_predict_calls, global_random_seed):
    test_iforest_works(contamination, global_random_seed)
    assert mocked_get_chunk.call_count == n_predict_calls