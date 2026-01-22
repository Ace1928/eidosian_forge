import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_axis_set_to_none_in_holoviews_opts_default(self, df, backend):
    hv.opts.defaults(hv.opts.Scatter(xaxis=None, yaxis=None))
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['xaxis'] is None
    assert opts.kwargs['yaxis'] is None