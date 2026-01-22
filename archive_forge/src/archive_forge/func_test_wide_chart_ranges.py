import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('line', Curve), ('area', Area), ('scatter', Scatter)])
def test_wide_chart_ranges(self, kind, element):
    plot = self.df.hvplot(kind=kind, xlim=(0, 3), ylim=(5, 10))
    opts = Store.lookup_options('bokeh', plot.last, 'plot').options
    self.assertEqual(opts['xlim'], (0, 3))
    self.assertEqual(opts['ylim'], (5, 10))