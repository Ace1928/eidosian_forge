import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def test_brier_score_loss():
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.9, 0.3, 1.0, 0.95])
    true_score = linalg.norm(y_true - y_pred) ** 2 / len(y_true)
    assert_almost_equal(brier_score_loss(y_true, y_true), 0.0)
    assert_almost_equal(brier_score_loss(y_true, y_pred), true_score)
    assert_almost_equal(brier_score_loss(1.0 + y_true, y_pred), true_score)
    assert_almost_equal(brier_score_loss(2 * y_true - 1, y_pred), true_score)
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred[1:])
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred + 1.0)
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred - 1.0)
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0.8, 0.6, 0.4, 0.2])
    error_message = 'Only binary classification is supported. The type of the target is multiclass'
    with pytest.raises(ValueError, match=error_message):
        brier_score_loss(y_true, y_pred)
    assert_almost_equal(brier_score_loss([-1], [0.4]), 0.16)
    assert_almost_equal(brier_score_loss([0], [0.4]), 0.16)
    assert_almost_equal(brier_score_loss([1], [0.4]), 0.36)
    assert_almost_equal(brier_score_loss(['foo'], [0.4], pos_label='bar'), 0.16)
    assert_almost_equal(brier_score_loss(['foo'], [0.4], pos_label='foo'), 0.36)