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
def test_averaging_binary_multilabel_all_zeroes():
    y_true = np.zeros((20, 3))
    y_pred = np.zeros((20, 3))
    y_true_binarize = y_true
    y_pred_binarize = y_pred
    binary_metric = lambda y_true, y_score, average='macro': _average_binary_score(precision_score, y_true, y_score, average)
    _check_averaging(binary_metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel=True)