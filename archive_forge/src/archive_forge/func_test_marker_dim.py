import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('kind', ['scatter', 'points'])
def test_marker_dim(self, df, kind, backend):
    plot = df.hvplot('x', 'y', marker='category', kind=kind)
    opts = Store.lookup_options(backend, plot, 'style')
    assert opts.kwargs['marker'] == 'category'
    assert 'category' in plot.vdims