import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('kind', ['scatter', 'points'])
def test_logz(self, df, kind, backend):
    plot = df.hvplot('x', 'y', c='x', logz=True, kind=kind)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['logz'] is True