import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('points', Points), ('paths', Path)])
def test_2d_set_hover_cols_to_all(self, kind, element):
    plot = self.cat_df.hvplot(x='x', y='y', hover_cols='all', kind=kind)
    data = plot.data[0] if kind == 'paths' else plot.data
    assert 'index' in data.columns
    self.assertEqual(plot, element(self.cat_df.reset_index(), ['x', 'y'], ['index', 'category']))