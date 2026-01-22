import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_default_y_not_in_by(self):
    plot = self.cat_df.hvplot.scatter(by='x')
    assert plot.kdims == ['x']
    assert plot[1].kdims == ['index']
    assert plot[1].vdims == ['y']