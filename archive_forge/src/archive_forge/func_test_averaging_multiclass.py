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
@pytest.mark.parametrize('name', sorted(METRICS_WITH_AVERAGING))
def test_averaging_multiclass(name):
    n_samples, n_classes = (50, 3)
    random_state = check_random_state(0)
    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    y_score = random_state.uniform(size=(n_samples, n_classes))
    lb = LabelBinarizer().fit(y_true)
    y_true_binarize = lb.transform(y_true)
    y_pred_binarize = lb.transform(y_pred)
    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)