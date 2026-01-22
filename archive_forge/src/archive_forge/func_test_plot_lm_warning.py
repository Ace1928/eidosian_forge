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
@pytest.mark.parametrize('warn_kwargs', [{'y_hat': 'bad_name'}, {'y_model': 'bad_name'}])
def test_plot_lm_warning(models, warn_kwargs):
    """Test Warning when needed groups or variables are not there in idata."""
    idata1 = models.model_1
    with pytest.warns(UserWarning):
        plot_lm(idata=from_dict(observed_data={'y': idata1.observed_data['y'].values}), y='y', **warn_kwargs)
    with pytest.warns(UserWarning):
        plot_lm(idata=idata1, y='y', **warn_kwargs)