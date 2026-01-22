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
@pytest.mark.parametrize('params', [{'input': ({'dashes': '-'}, 'scatter'), 'output': 'linestyle'}, {'input': ({'mfc': 'blue', 'c': 'blue', 'line_width': 2}, 'plot'), 'output': ('markerfacecolor', 'color', 'line_width')}, {'input': ({'ec': 'blue', 'fc': 'black'}, 'hist'), 'output': ('edgecolor', 'facecolor')}, {'input': ({'edgecolors': 'blue', 'lw': 3}, 'hlines'), 'output': ('edgecolor', 'linewidth')}])
def test_matplotlib_kwarg_dealiaser(params):
    dealiased = matplotlib_kwarg_dealiaser(params['input'][0], kind=params['input'][1])
    for returned in dealiased:
        assert returned in params['output']