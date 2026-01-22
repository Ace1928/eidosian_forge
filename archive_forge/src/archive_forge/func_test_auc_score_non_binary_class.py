import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
def test_auc_score_non_binary_class():
    rng = check_random_state(404)
    y_pred = rng.rand(10)
    y_true = np.zeros(10, dtype='int')
    err_msg = 'ROC AUC score is not defined'
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.ones(10, dtype='int')
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.full(10, -1, dtype='int')
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    with warnings.catch_warnings(record=True):
        rng = check_random_state(404)
        y_pred = rng.rand(10)
        y_true = np.zeros(10, dtype='int')
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        y_true = np.ones(10, dtype='int')
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        y_true = np.full(10, -1, dtype='int')
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)