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
@pytest.mark.parametrize('name', sorted(MULTILABELS_METRICS - {'unnormalized_multilabel_confusion_matrix'}))
def test_multilabel_label_permutations_invariance(name):
    random_state = check_random_state(0)
    n_samples, n_classes = (20, 4)
    y_true = random_state.randint(0, 2, size=(n_samples, n_classes))
    y_score = random_state.randint(0, 2, size=(n_samples, n_classes))
    metric = ALL_METRICS[name]
    score = metric(y_true, y_score)
    for perm in permutations(range(n_classes), n_classes):
        y_score_perm = y_score[:, perm]
        y_true_perm = y_true[:, perm]
        current_score = metric(y_true_perm, y_score_perm)
        assert_almost_equal(score, current_score)