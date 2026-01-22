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
def test_plot_posterior_skipna_combinedims():
    idata = load_arviz_data('centered_eight')
    idata.posterior['theta'].loc[dict(school='Deerfield')] = np.nan
    with pytest.raises(ValueError):
        plot_posterior(idata, var_names='theta', combine_dims={'school'}, skipna=False)
    ax = plot_posterior(idata, var_names='theta', combine_dims={'school'}, skipna=True)
    assert not isinstance(ax, np.ndarray)