from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize('metric_name', chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
def test_permute_labels(metric_name):
    y_label = np.array([0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0])
    if metric_name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[metric_name]
        score_1 = metric(y_pred, y_label)
        assert_allclose(score_1, metric(1 - y_pred, y_label))
        assert_allclose(score_1, metric(1 - y_pred, 1 - y_label))
        assert_allclose(score_1, metric(y_pred, 1 - y_label))
    else:
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(7, 10))
        score_1 = metric(X, y_pred)
        assert_allclose(score_1, metric(X, 1 - y_pred))