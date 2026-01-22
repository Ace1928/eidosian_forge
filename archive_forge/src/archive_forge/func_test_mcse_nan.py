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
@pytest.mark.parametrize('mcse_method', ('mean', 'sd', 'median', 'quantile'))
@pytest.mark.parametrize('chain', (None, 1, 2))
@pytest.mark.parametrize('draw', (1, 2, 3, 4))
@pytest.mark.parametrize('use_nan', (True, False))
def test_mcse_nan(self, mcse_method, chain, draw, use_nan):
    data = np.random.randn(draw) if chain is None else np.random.randn(chain, draw)
    if use_nan:
        data[0] = np.nan
    if mcse_method == 'quantile':
        mcse_hat = mcse(data, method=mcse_method, prob=0.34)
    else:
        mcse_hat = mcse(data, method=mcse_method)
    if draw < 4 or use_nan:
        assert np.isnan(mcse_hat)
    else:
        assert not np.isnan(mcse_hat)