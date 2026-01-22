from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize('metric_name', NORMALIZED_METRICS)
def test_normalized_output(metric_name):
    upper_bound_1 = [0, 0, 0, 1, 1, 1]
    upper_bound_2 = [0, 0, 0, 1, 1, 1]
    metric = SUPERVISED_METRICS[metric_name]
    assert metric([0, 0, 0, 1, 1], [0, 0, 0, 1, 2]) > 0.0
    assert metric([0, 0, 1, 1, 2], [0, 0, 1, 1, 1]) > 0.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric(upper_bound_1, upper_bound_2) == pytest.approx(1.0)
    lower_bound_1 = [0, 0, 0, 0, 0, 0]
    lower_bound_2 = [0, 1, 2, 3, 4, 5]
    score = np.array([metric(lower_bound_1, lower_bound_2), metric(lower_bound_2, lower_bound_1)])
    assert not (score < 0).any()