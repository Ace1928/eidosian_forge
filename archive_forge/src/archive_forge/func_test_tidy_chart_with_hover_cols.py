import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('line', Curve), ('area', Area), ('scatter', Scatter)])
def test_tidy_chart_with_hover_cols(self, kind, element):
    plot = self.cat_df.hvplot(x='x', y='y', kind=kind, hover_cols=['category'])
    self.assertEqual(plot, element(self.cat_df, 'x', ['y', 'category']))