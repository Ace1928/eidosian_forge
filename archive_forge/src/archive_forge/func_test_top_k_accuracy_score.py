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
@pytest.mark.parametrize('y_true, k, true_score', [([0, 1, 2, 3], 1, 0.25), ([0, 1, 2, 3], 2, 0.5), ([0, 1, 2, 3], 3, 0.75)])
def test_top_k_accuracy_score(y_true, k, true_score):
    y_score = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.3, 0.4, 0.2], [0.4, 0.1, 0.2, 0.3], [0.3, 0.2, 0.4, 0.1]])
    score = top_k_accuracy_score(y_true, y_score, k=k)
    assert score == pytest.approx(true_score)