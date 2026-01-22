import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from pandas import DataFrame
from scipy.stats import gaussian_kde, norm
import xarray as xr
from ...data import from_dict, load_arviz_data
from ...plots import (
from ...rcparams import rc_context, rcParams
from ...stats import compare, hdi, loo, waic
from ...stats.density_utils import kde as _kde
from ...utils import _cov
from ...plots.plot_utils import plot_point_interval
from ...plots.dotplot import wilkinson_algorithm
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.parametrize('kwargs', [{}, {'y_hat': 'bad_name'}, {'x': 'x'}, {'x': ('x', 'x')}, {'y_holdout': 'z'}, {'y_holdout': 'z', 'x_holdout': 'x_pred'}, {'x': ('x', 'x'), 'y_holdout': 'z', 'x_holdout': ('x_pred', 'x_pred')}, {'y_forecasts': 'z'}, {'y_holdout': 'z', 'y_forecasts': 'bad_name'}])
def test_plot_ts(kwargs):
    """Test timeseries plots basic functionality."""
    nchains = 4
    ndraws = 500
    obs_data = {'y': 2 * np.arange(1, 9) + 3, 'z': 2 * np.arange(8, 12) + 3}
    posterior_predictive = {'y': np.random.normal(obs_data['y'] * 1.2 - 3, size=(nchains, ndraws, len(obs_data['y']))), 'z': np.random.normal(obs_data['z'] * 1.2 - 3, size=(nchains, ndraws, len(obs_data['z'])))}
    const_data = {'x': np.arange(1, 9), 'x_pred': np.arange(8, 12)}
    idata = from_dict(observed_data=obs_data, posterior_predictive=posterior_predictive, constant_data=const_data, coords={'obs_dim': np.arange(1, 9), 'pred_dim': np.arange(8, 12)}, dims={'y': ['obs_dim'], 'z': ['pred_dim']})
    ax = plot_ts(idata=idata, y='y', **kwargs)
    assert np.all(ax)