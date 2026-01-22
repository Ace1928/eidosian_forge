import copy
import itertools
import pickle
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, create_memmap_backed_data
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version, sp_version
@pytest.mark.parametrize('metric, metric_kwargs', METRICS_DEFAULT_PARAMS)
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_get_metric_dtype(metric, metric_kwargs, dtype):
    specialized_cls = {np.float32: DistanceMetric32, np.float64: DistanceMetric64}[dtype]
    metric_kwargs = {k: v[0] for k, v in metric_kwargs.items()}
    generic_type = type(DistanceMetric.get_metric(metric, dtype, **metric_kwargs))
    specialized_type = type(specialized_cls.get_metric(metric, **metric_kwargs))
    assert generic_type is specialized_type