import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_scatter_size_set_to_series(self):
    if is_dask(self.df['y']):
        y = self.df['y'].compute()
    else:
        y = self.df['y']
    plot = self.df.hvplot.scatter('x', 'y', s=y)
    opts = Store.lookup_options('bokeh', plot, 'style')
    assert '_size' in plot.data.columns
    self.assertEqual(opts.kwargs['size'], '_size')