import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_hvplot_defaults(self, df, backend):
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'plot')
    if backend == 'bokeh':
        assert opts.kwargs['height'] == 300
        assert opts.kwargs['width'] == 700
    elif backend == 'matplotlib':
        assert opts.kwargs['aspect'] == pytest.approx(2.333333)
        assert opts.kwargs['fig_size'] == pytest.approx(233.333333)
    if backend == 'bokeh':
        assert opts.kwargs['responsive'] is False
        assert opts.kwargs['shared_axes'] is True
        assert opts.kwargs['legend_position'] == 'right'
    assert opts.kwargs['show_grid'] is False
    assert opts.kwargs['show_legend'] is True
    assert opts.kwargs['logx'] is False
    assert opts.kwargs['logy'] is False
    assert opts.kwargs.get('logz') is None