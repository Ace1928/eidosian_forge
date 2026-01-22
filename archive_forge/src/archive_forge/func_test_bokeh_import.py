import importlib
import numpy as np
import pytest
import xarray as xr
from ...data import from_dict
from ...plots.backends.matplotlib import dealiase_sel_kwargs, matplotlib_kwarg_dealiaser
from ...plots.plot_utils import (
from ...rcparams import rc_context
from ...sel_utils import xarray_sel_iter, xarray_to_ndarray
from ...stats.density_utils import get_bins
from ...utils import get_coords
from ..helpers import running_on_ci
@pytest.mark.skipif(not (bokeh_installed or running_on_ci()), reason='test requires bokeh which is not installed')
def test_bokeh_import():
    """Tests that correct method is returned on bokeh import"""
    plot = get_plotting_function('plot_dist', 'distplot', 'bokeh')
    from ...plots.backends.bokeh.distplot import plot_dist
    assert plot is plot_dist