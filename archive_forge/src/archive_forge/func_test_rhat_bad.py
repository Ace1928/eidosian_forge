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
def test_rhat_bad(self):
    """Confirm rank normalized Split R-hat statistic is
        far from 1 for a small number of samples."""
    r_hat = rhat(np.vstack([20 + np.random.randn(1, 100), np.random.randn(1, 100)]))
    assert 1 / GOOD_RHAT > r_hat or GOOD_RHAT < r_hat