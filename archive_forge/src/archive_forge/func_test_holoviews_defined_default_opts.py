import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('backend', ['bokeh', pytest.param('matplotlib', marks=pytest.mark.xfail(reason='default opts not supported w/ matplotlib')), pytest.param('plotly', marks=pytest.mark.xfail(reason='default opts not supported w/ plotly'))], indirect=True)
def test_holoviews_defined_default_opts(self, df, backend):
    hv.opts.defaults(hv.opts.Scatter(height=400, width=900, show_grid=True))
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'plot')
    if backend == 'bokeh':
        assert opts.kwargs['legend_position'] == 'right'
    assert opts.kwargs['show_grid'] is True
    assert opts.kwargs['height'] == 400
    assert opts.kwargs['width'] == 900