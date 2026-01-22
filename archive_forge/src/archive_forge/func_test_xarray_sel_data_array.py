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
def test_xarray_sel_data_array(sample_dataset):
    """Assert that varname order stays consistent when chains are combined

    Touches code that is hard to reach.
    """
    _, _, data = sample_dataset
    var_names = [var for var, _, _ in xarray_sel_iter(data.mu, var_names=None, combined=True)]
    assert set(var_names) == {'mu'}