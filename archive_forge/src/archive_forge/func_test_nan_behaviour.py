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
@pytest.mark.parametrize('func', ('_mcse_quantile', '_z_scale'))
def test_nan_behaviour(self, func):
    data = np.random.randn(100, 4)
    data[0, 0] = np.nan
    if func == '_mcse_quantile':
        assert np.isnan(_mcse_quantile(data, 0.5)).all(None)
    elif packaging.version.parse(scipy.__version__) < packaging.version.parse('1.10.0.dev0'):
        assert not np.isnan(_z_scale(data)).all(None)
        assert not np.isnan(_z_scale(data)).any(None)
    else:
        assert np.isnan(_z_scale(data)).sum() == 1