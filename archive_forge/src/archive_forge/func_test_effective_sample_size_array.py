import os
import numpy as np
import packaging
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_almost_equal
from ...data import from_cmdstan, load_arviz_data
from ...rcparams import rcParams
from ...sel_utils import xarray_var_iter
from ...stats import bfmi, ess, mcse, rhat
from ...stats.diagnostics import (
@pytest.mark.parametrize('method', ('bulk', 'tail', 'quantile', 'local', 'mean', 'sd', 'median', 'mad', 'z_scale', 'folded', 'identity'))
@pytest.mark.parametrize('relative', (True, False))
def test_effective_sample_size_array(self, data, method, relative):
    n_low = 100 / 400 if relative else 100
    n_high = 800 / 400 if relative else 800
    if method in ('quantile', 'tail'):
        ess_hat = ess(data, method=method, prob=0.34, relative=relative)
        if method == 'tail':
            assert ess_hat > n_low
            assert ess_hat < n_high
            ess_hat = ess(np.random.randn(4, 100), method=method, relative=relative)
            assert ess_hat > n_low
            assert ess_hat < n_high
            ess_hat = ess(np.random.randn(4, 100), method=method, prob=(0.2, 0.8), relative=relative)
    elif method == 'local':
        ess_hat = ess(np.random.randn(4, 100), method=method, prob=(0.2, 0.3), relative=relative)
    else:
        ess_hat = ess(np.random.randn(4, 100), method=method, relative=relative)
    assert ess_hat > n_low
    assert ess_hat < n_high