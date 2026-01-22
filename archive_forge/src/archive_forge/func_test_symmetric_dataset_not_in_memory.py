import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_symmetric_dataset_not_in_memory(self):
    da = xr.DataArray(data=np.arange(-100, 100).reshape(10, 10, 2), dims=['x', 'y', 'z'], coords={'x': np.arange(10), 'y': np.arange(10), 'z': np.arange(2)})
    ds = xr.Dataset(data_vars={'value': da})
    with tempfile.TemporaryDirectory() as tempdir:
        fpath = os.path.join(tempdir, 'data.nc')
        ds.to_netcdf(fpath)
        ds = xr.open_dataset(fpath)
        plot = ds.value.hvplot(x='x', y='y', check_symmetric_max=ds.value.size + 1)
        plot[0]
        plot_opts = Store.lookup_options('bokeh', plot.last, 'plot')
        assert not plot_opts.kwargs['symmetric']
        ds.close()