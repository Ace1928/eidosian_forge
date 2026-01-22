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
@pytest.mark.parametrize('metric_param_grid', METRICS_DEFAULT_PARAMS, ids=lambda params: params[0])
@pytest.mark.parametrize('X, Y', [(X64, Y64), (X32, Y32), (X_mmap, Y_mmap)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_cdist(metric_param_grid, X, Y, csr_container):
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    X_csr, Y_csr = (csr_container(X), csr_container(Y))
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        rtol_dict = {}
        if metric == 'mahalanobis' and X.dtype == np.float32:
            rtol_dict = {'rtol': 1e-06}
        if metric == 'minkowski':
            p = kwargs['p']
            if sp_version < parse_version('1.7.0') and p < 1:
                pytest.skip('scipy does not support 0<p<1 for minkowski metric < 1.7.0')
        D_scipy_cdist = cdist(X, Y, metric, **kwargs)
        dm = DistanceMetric.get_metric(metric, X.dtype, **kwargs)
        D_sklearn = dm.pairwise(X, Y)
        assert D_sklearn.flags.c_contiguous
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)
        D_sklearn = dm.pairwise(X_csr, Y_csr)
        assert D_sklearn.flags.c_contiguous
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)
        D_sklearn = dm.pairwise(X_csr, Y)
        assert D_sklearn.flags.c_contiguous
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)
        D_sklearn = dm.pairwise(X, Y_csr)
        assert D_sklearn.flags.c_contiguous
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)