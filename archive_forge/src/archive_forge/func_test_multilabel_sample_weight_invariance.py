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
@pytest.mark.parametrize('name', sorted((MULTILABELS_METRICS | THRESHOLDED_MULTILABEL_METRICS | MULTIOUTPUT_METRICS) - METRICS_WITHOUT_SAMPLE_WEIGHT))
def test_multilabel_sample_weight_invariance(name):
    random_state = check_random_state(0)
    _, ya = make_multilabel_classification(n_features=1, n_classes=10, random_state=0, n_samples=50, allow_unlabeled=False)
    _, yb = make_multilabel_classification(n_features=1, n_classes=10, random_state=1, n_samples=50, allow_unlabeled=False)
    y_true = np.vstack([ya, yb])
    y_pred = np.vstack([ya, ya])
    y_score = random_state.randint(1, 4, size=y_true.shape)
    metric = ALL_METRICS[name]
    if name in THRESHOLDED_METRICS:
        check_sample_weight_invariance(name, metric, y_true, y_score)
    else:
        check_sample_weight_invariance(name, metric, y_true, y_pred)