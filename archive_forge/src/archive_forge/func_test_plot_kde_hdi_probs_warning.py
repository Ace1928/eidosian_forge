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
@pytest.mark.parametrize('kwargs', [{'hdi_probs': [0.3, 0.6, 0.9], 'contourf_kwargs': {'levels': [0, 0.5, 1]}}, {'hdi_probs': [0.3, 0.6, 0.9], 'contour_kwargs': {'levels': [0, 0.5, 1]}}])
def test_plot_kde_hdi_probs_warning(continuous_model, kwargs):
    """Ensure warning is raised when too many keywords are specified."""
    with pytest.warns(UserWarning):
        axes = plot_kde(continuous_model['x'], continuous_model['y'], **kwargs)
    assert axes