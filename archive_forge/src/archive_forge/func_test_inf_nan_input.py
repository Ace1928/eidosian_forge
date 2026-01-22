from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('metric_name, metric_func', dict(SUPERVISED_METRICS, **UNSUPERVISED_METRICS).items())
def test_inf_nan_input(metric_name, metric_func):
    if metric_name in SUPERVISED_METRICS:
        invalids = [([0, 1], [np.inf, np.inf]), ([0, 1], [np.nan, np.nan]), ([0, 1], [np.nan, np.inf])]
    else:
        X = np.random.randint(10, size=(2, 10))
        invalids = [(X, [np.inf, np.inf]), (X, [np.nan, np.nan]), (X, [np.nan, np.inf])]
    with pytest.raises(ValueError, match='contains (NaN|infinity)'):
        for args in invalids:
            metric_func(*args)