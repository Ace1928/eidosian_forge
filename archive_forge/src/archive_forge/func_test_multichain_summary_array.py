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
@pytest.mark.parametrize('draws', (3, 4, 100))
@pytest.mark.parametrize('chains', (None, 1, 2))
def test_multichain_summary_array(self, draws, chains):
    """Test multichain statistics against individual functions."""
    if chains is None:
        ary = np.random.randn(draws)
    else:
        ary = np.random.randn(chains, draws)
    mcse_mean_hat = mcse(ary, method='mean')
    mcse_sd_hat = mcse(ary, method='sd')
    ess_bulk_hat = ess(ary, method='bulk')
    ess_tail_hat = ess(ary, method='tail')
    rhat_hat = _rhat_rank(ary)
    mcse_mean_hat_, mcse_sd_hat_, ess_bulk_hat_, ess_tail_hat_, rhat_hat_ = _multichain_statistics(ary)
    if draws == 3:
        assert np.isnan((mcse_mean_hat, mcse_sd_hat, ess_bulk_hat, ess_tail_hat, rhat_hat)).all()
        assert np.isnan((mcse_mean_hat_, mcse_sd_hat_, ess_bulk_hat_, ess_tail_hat_, rhat_hat_)).all()
    else:
        assert_almost_equal(mcse_mean_hat, mcse_mean_hat_)
        assert_almost_equal(mcse_sd_hat, mcse_sd_hat_)
        assert_almost_equal(ess_bulk_hat, ess_bulk_hat_)
        assert_almost_equal(ess_tail_hat, ess_tail_hat_)
        if chains in (None, 1):
            assert np.isnan(rhat_hat)
            assert np.isnan(rhat_hat_)
        else:
            assert round(rhat_hat, 3) == round(rhat_hat_, 3)