import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('kind', ['scatter', 'points'])
def test_size_dim_overlay(self, df, kind, backend):
    plot = df.hvplot('x', 'y', s='number', by='category', kind=kind)
    opts = Store.lookup_options(backend, plot.last, 'style')
    if backend in ['bokeh', 'plotly']:
        param = 'size'
    elif backend == 'matplotlib':
        param = 's'
    assert opts.kwargs[param] == 'number'
    assert 'number' in plot.last.vdims