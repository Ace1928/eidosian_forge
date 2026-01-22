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
@pytest.mark.parametrize('kwargs', [{'plot_kwargs': {'linestyle': '-'}}, {'contour': True, 'fill_last': False}, {'contour': False}])
def test_plot_dist_2d_kde(continuous_model, kwargs):
    axes = plot_dist(continuous_model['x'], continuous_model['y'], **kwargs)
    assert axes