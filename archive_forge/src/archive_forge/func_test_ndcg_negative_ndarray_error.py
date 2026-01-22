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
def test_ndcg_negative_ndarray_error():
    """Check `ndcg_score` exception when `y_true` contains negative values."""
    y_true = np.array([[-0.89, -0.53, -0.47, 0.39, 0.56]])
    y_score = np.array([[0.07, 0.31, 0.75, 0.33, 0.27]])
    expected_message = 'ndcg_score should not be used on negative y_true values'
    with pytest.raises(ValueError, match=expected_message):
        ndcg_score(y_true, y_score)