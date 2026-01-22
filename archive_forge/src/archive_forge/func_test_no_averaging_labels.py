from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@ignore_warnings
def test_no_averaging_labels():
    y_true_multilabel = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    y_pred_multilabel = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
    y_true_multiclass = np.array([0, 1, 2])
    y_pred_multiclass = np.array([0, 2, 3])
    labels = np.array([3, 0, 1, 2])
    _, inverse_labels = np.unique(labels, return_inverse=True)
    for name in METRICS_WITH_AVERAGING:
        for y_true, y_pred in [[y_true_multiclass, y_pred_multiclass], [y_true_multilabel, y_pred_multilabel]]:
            if name not in MULTILABELS_METRICS and y_pred.ndim > 1:
                continue
            metric = ALL_METRICS[name]
            score_labels = metric(y_true, y_pred, labels=labels, average=None)
            score = metric(y_true, y_pred, average=None)
            assert_array_equal(score_labels, score[inverse_labels])