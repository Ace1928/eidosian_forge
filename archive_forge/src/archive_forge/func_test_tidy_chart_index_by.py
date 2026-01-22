import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('line', Curve), ('area', Area), ('scatter', Scatter)])
def test_tidy_chart_index_by(self, kind, element):
    plot = self.df.hvplot(x='index', y='y', by='x', kind=kind)
    obj = NdOverlay({1: element(self.df[self.df.x == 1], 'index', 'y'), 3: element(self.df[self.df.x == 3], 'index', 'y'), 5: element(self.df[self.df.x == 5], 'index', 'y')}, 'x')
    self.assertEqual(plot, obj)