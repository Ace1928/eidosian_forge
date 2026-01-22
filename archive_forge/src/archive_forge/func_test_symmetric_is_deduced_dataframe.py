import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_symmetric_is_deduced_dataframe(self, symmetric_df, backend):
    plot = symmetric_df.hvplot.scatter('x', 'y', c='number')
    plot_opts = Store.lookup_options(backend, plot, 'plot')
    assert plot_opts.kwargs['symmetric'] is True
    style_opts = Store.lookup_options(backend, plot, 'style')
    assert style_opts.kwargs['cmap'] == 'coolwarm'