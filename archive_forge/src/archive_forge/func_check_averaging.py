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
def check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score):
    is_multilabel = type_of_target(y_true).startswith('multilabel')
    metric = ALL_METRICS[name]
    if name in METRICS_WITH_AVERAGING:
        _check_averaging(metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel)
    elif name in THRESHOLDED_METRICS_WITH_AVERAGING:
        _check_averaging(metric, y_true, y_score, y_true_binarize, y_score, is_multilabel)
    else:
        raise ValueError('Metric is not recorded as having an average option')