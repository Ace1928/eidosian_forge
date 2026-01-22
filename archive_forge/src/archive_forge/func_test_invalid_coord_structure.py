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
def test_invalid_coord_structure(self, sample_dataset):
    """Assert that nicer exception appears when user enters wrong coords datatype"""
    _, _, data = sample_dataset
    coords = {'draw'}
    with pytest.raises(TypeError):
        get_coords(data, coords)