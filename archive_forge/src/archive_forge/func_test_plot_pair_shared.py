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
@pytest.mark.parametrize('sharex', ['col', None])
@pytest.mark.parametrize('sharey', ['row', None])
@pytest.mark.parametrize('marginals', [True, False])
def test_plot_pair_shared(sharex, sharey, marginals):
    rng = np.random.default_rng()
    idata = from_dict({'a': rng.standard_normal((4, 500, 5))})
    numvars = 5 - (not marginals)
    if sharex is None and sharey is None:
        ax = plot_pair(idata, marginals=marginals)
    else:
        backend_kwargs = {}
        if sharex is not None:
            backend_kwargs['sharex'] = sharex
        if sharey is not None:
            backend_kwargs['sharey'] = sharey
        with pytest.warns(UserWarning):
            ax = plot_pair(idata, marginals=marginals, backend_kwargs=backend_kwargs)
    for i in range(numvars):
        num_shared_x = numvars - i
        assert len(ax[-1, i].get_shared_x_axes().get_siblings(ax[-1, i])) == num_shared_x
    for j in range(numvars):
        if marginals:
            num_shared_y = j
            assert len(ax[j, j].get_shared_y_axes().get_siblings(ax[j, j])) == 1
            if j == 0:
                continue
        else:
            num_shared_y = j + 1
        assert len(ax[j, 0].get_shared_y_axes().get_siblings(ax[j, 0])) == num_shared_y