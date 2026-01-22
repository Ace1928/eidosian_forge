import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_heatmap_2d_derived_x_and_y(self):
    plot = self.time_df.hvplot.heatmap(x='time.hour', y='time.day', C='temp')
    assert plot.kdims == ['time.hour', 'time.day']
    assert plot.vdims == ['temp']