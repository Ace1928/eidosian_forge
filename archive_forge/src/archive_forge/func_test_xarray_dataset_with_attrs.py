import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_xarray_dataset_with_attrs(self):
    try:
        import xarray as xr
        import hvplot.xarray
    except ImportError:
        raise SkipTest('xarray not available')
    dset = xr.Dataset({'u': ('t', [1, 3]), 'v': ('t', [4, 2])}, coords={'t': ('t', [0, 1], {'long_name': 'time', 'units': 's'})})
    ndoverlay = dset.hvplot.line()
    assert render(ndoverlay, 'bokeh').xaxis.axis_label == 'time (s)'