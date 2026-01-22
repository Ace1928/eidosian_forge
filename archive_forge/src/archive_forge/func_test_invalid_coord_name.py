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
def test_invalid_coord_name(self, sample_dataset):
    """Assert that nicer exception appears when user enters wrong coords name"""
    _, _, data = sample_dataset
    coords = {'NOT_A_COORD_NAME': [1]}
    with pytest.raises((KeyError, ValueError), match="Coords ({'NOT_A_COORD_NAME'} are invalid coordinate keys|should follow mapping format {coord_name:\\[dim1, dim2\\]})"):
        get_coords(data, coords)