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
def check_zero_or_all_relevant_labels(lrap_score):
    random_state = check_random_state(0)
    for n_labels in range(2, 5):
        y_score = random_state.uniform(size=(1, n_labels))
        y_score_ties = np.zeros_like(y_score)
        y_true = np.zeros((1, n_labels))
        assert lrap_score(y_true, y_score) == 1.0
        assert lrap_score(y_true, y_score_ties) == 1.0
        y_true = np.ones((1, n_labels))
        assert lrap_score(y_true, y_score) == 1.0
        assert lrap_score(y_true, y_score_ties) == 1.0
    assert_almost_equal(lrap_score([[1], [0], [1], [0]], [[0.5], [0.5], [0.5], [0.5]]), 1.0)