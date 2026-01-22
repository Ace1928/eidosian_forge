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
@pytest.mark.parametrize('var_names', (None, 'mu', ['mu', 'tau']))
def test_effective_sample_size_dataset(self, data, method, var_names, relative):
    n_low = 100 / (data.chain.size * data.draw.size) if relative else 100
    if method in ('quantile', 'tail'):
        ess_hat = ess(data, var_names=var_names, method=method, prob=0.34, relative=relative)
    elif method == 'local':
        ess_hat = ess(data, var_names=var_names, method=method, prob=(0.2, 0.3), relative=relative)
    else:
        ess_hat = ess(data, var_names=var_names, method=method, relative=relative)
    assert np.all(ess_hat.mu.values > n_low)