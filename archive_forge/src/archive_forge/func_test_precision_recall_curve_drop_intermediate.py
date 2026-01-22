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
def test_precision_recall_curve_drop_intermediate():
    """Check the behaviour of the `drop_intermediate` parameter."""
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    assert_allclose(thresholds, [0.0, 0.7, 1.0])
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    assert_allclose(thresholds, [0.0, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_true = [0, 0, 0, 0]
    y_score = [0.0, 0.1, 0.2, 0.3]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    assert_allclose(thresholds, [0.0, 0.3])
    y_true = [1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.2, 0.3]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    assert_allclose(thresholds, [0.0, 0.1, 0.2, 0.3])