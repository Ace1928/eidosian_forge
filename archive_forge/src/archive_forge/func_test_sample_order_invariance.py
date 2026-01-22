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
@pytest.mark.parametrize('name', sorted(set(ALL_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS))
def test_sample_order_invariance(name):
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))
    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)
    y_true_shuffle, y_pred_shuffle = shuffle(y_true, y_pred, random_state=0)
    with ignore_warnings():
        metric = ALL_METRICS[name]
        assert_allclose(metric(y_true, y_pred), metric(y_true_shuffle, y_pred_shuffle), err_msg='%s is not sample order invariant' % name)