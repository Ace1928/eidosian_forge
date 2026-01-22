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
def test_set_bokeh_circular_ticks_labels():
    """Assert the axes returned after placing ticks and tick labels for circular plots."""
    import bokeh.plotting as bkp
    ax = bkp.figure(x_axis_type=None, y_axis_type=None)
    hist = np.linspace(0, 1, 10)
    labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    ax = set_bokeh_circular_ticks_labels(ax, hist, labels)
    renderers = ax.renderers
    assert len(renderers) == 3
    assert renderers[2].data_source.data['text'] == labels
    assert len(renderers[0].data_source.data['start_angle']) == len(labels)