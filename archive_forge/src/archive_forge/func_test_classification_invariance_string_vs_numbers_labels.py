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
@pytest.mark.parametrize('name', sorted(set(CLASSIFICATION_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS))
def test_classification_invariance_string_vs_numbers_labels(name):
    random_state = check_random_state(0)
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))
    y1_str = np.array(['eggs', 'spam'])[y1]
    y2_str = np.array(['eggs', 'spam'])[y2]
    pos_label_str = 'spam'
    labels_str = ['eggs', 'spam']
    with ignore_warnings():
        metric = CLASSIFICATION_METRICS[name]
        measure_with_number = metric(y1, y2)
        metric_str = metric
        if name in METRICS_WITH_POS_LABEL:
            metric_str = partial(metric_str, pos_label=pos_label_str)
        measure_with_str = metric_str(y1_str, y2_str)
        assert_array_equal(measure_with_number, measure_with_str, err_msg='{0} failed string vs number invariance test'.format(name))
        measure_with_strobj = metric_str(y1_str.astype('O'), y2_str.astype('O'))
        assert_array_equal(measure_with_number, measure_with_strobj, err_msg='{0} failed string object vs number invariance test'.format(name))
        if name in METRICS_WITH_LABELS:
            metric_str = partial(metric_str, labels=labels_str)
            measure_with_str = metric_str(y1_str, y2_str)
            assert_array_equal(measure_with_number, measure_with_str, err_msg='{0} failed string vs number  invariance test'.format(name))
            measure_with_strobj = metric_str(y1_str.astype('O'), y2_str.astype('O'))
            assert_array_equal(measure_with_number, measure_with_strobj, err_msg='{0} failed string vs number  invariance test'.format(name))