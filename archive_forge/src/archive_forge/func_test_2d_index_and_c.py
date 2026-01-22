import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
@parameterized.expand([('points', Points), ('paths', Path)])
def test_2d_index_and_c(self, kind, element):
    plot = self.df.hvplot(x='index', y='y', c='x', kind=kind)
    self.assertEqual(plot, element(self.df, ['index', 'y'], ['x']))