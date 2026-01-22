import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_symmetric_dataframe(self, backend):
    import pandas as pd
    df = pd.DataFrame([[1, 2, -1], [3, 4, 0], [5, 6, 1]], columns=['x', 'y', 'number'])
    plot = df.hvplot.scatter('x', 'y', c='number')
    plot_opts = Store.lookup_options(backend, plot, 'plot')
    assert plot_opts.kwargs['symmetric'] is True
    style_opts = Store.lookup_options(backend, plot, 'style')
    assert style_opts.kwargs['cmap'] == 'coolwarm'