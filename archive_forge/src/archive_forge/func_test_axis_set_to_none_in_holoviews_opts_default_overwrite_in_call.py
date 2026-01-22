import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.xfail
def test_axis_set_to_none_in_holoviews_opts_default_overwrite_in_call(self, df, backend):
    hv.opts.defaults(hv.opts.Scatter(xaxis=None, yaxis=None))
    plot = df.hvplot.scatter('x', 'y', c='category', xaxis=True, yaxis=True)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert 'xaxis' not in opts.kwargs
    assert 'yaxis' not in opts.kwargs