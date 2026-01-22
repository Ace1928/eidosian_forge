import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('line', Curve), ('area', Area), ('scatter', Scatter)])
def test_wide_chart_legend_position(self, kind, element):
    plot = self.df.hvplot(kind=kind, value_label='Test', group_label='Category', legend='left')
    opts = Store.lookup_options('bokeh', plot, 'plot')
    self.assertEqual(opts.kwargs['legend_position'], 'left')