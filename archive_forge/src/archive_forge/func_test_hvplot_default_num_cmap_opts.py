import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_hvplot_default_num_cmap_opts(self, df, backend):
    plot = df.hvplot.scatter('x', 'y', c='number')
    opts = Store.lookup_options(backend, plot, 'style')
    assert opts.kwargs['cmap'] == 'kbc_r'