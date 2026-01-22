import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('backend', ['bokeh', pytest.param('matplotlib', marks=pytest.mark.xfail(reason='default opts not supported not supported w/ matplotlib')), pytest.param('plotly', marks=pytest.mark.xfail(reason='default opts not supported not supported w/ plotly'))], indirect=True)
def test_holoviews_defined_default_opts_are_not_mutable(self, df, backend):
    hv.opts.defaults(hv.opts.Scatter(tools=['tap']))
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['tools'] == ['tap', 'hover']
    default_opts = Store.options(backend=backend)['Scatter'].groups['plot'].options
    assert default_opts['tools'] == ['tap']